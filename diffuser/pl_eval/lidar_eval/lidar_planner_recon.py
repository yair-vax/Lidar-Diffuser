"""
Option B -- planner LiDAR-reconstruction dump (-> --recon_npz for the histogram eval).

Headline LiDAR result: show that the maze locations whose LiDAR signature is
*ambiguous* (Option A's ambiguity map) are exactly where the trained **planner**
reconstructs scans worst -- i.e. maze structure predicts model difficulty.

This script produces the ``recon_npz`` that ``lidar_pos_histogram.py`` consumes via
``--recon_npz`` (keys ``true_xy`` (N,2), ``true_lidar`` (N,30), ``pred_lidar`` (N,30)).
The histogram script then bins per-transition LiDAR RMSE onto maze cells and reports
``corr_ambiguity_vs_error`` + a scatter.

Reconstruction protocol (the denoising protocol from LIDAR_ROLLOUT/NEXT_STEPS §3):
the planner is trained to reconstruct clean LiDAR sequences (``predict_epsilon=False``
-> it predicts x_0). For each held-out LiDAR chunk x_start (normalized, [B,H,30]):

    t       = a fixed mid-level diffusion timestep (default T//4; can average a few)
    x_noisy = model.q_sample(x_start, t, noise)
    # condition on the chunk's own start/goal scans (matches how the model is used):
    x_cond, tj_cond = model.get_tj_cond(x_noisy, {do_cond:'both_stgl', stgl_cond:{0,H-1}}, t)
    x_recon = model.p_mean_variance(x_cond, t, tj_cond, return_modelout=True)[3]   # pred x_0
    pred_lidar = unnormalize(x_recon);  true_lidar = unnormalize(x_start)

The two inpainted endpoints (steps 0 and H-1) are excluded from the dump (they are
conditioned, so trivially reconstructed). The held-out split matches the histogram
script's ``split_episodes`` exactly, so A's ambiguity map and B's error map line up.

Needs GPU + the LiDAR planner checkpoint + a full LiDAR npz (from gen_lidar_dataset.py).
No MuJoCo, no inv-dyn.

Example::

    # 1) materialize the full LiDAR dataset (shared with Option A):
    python -m diffuser.datasets.lidar.gen_lidar_dataset \
        --env antmaze-giant-stitch-v0 \
        --out data/ogb_maze/lidar/antmaze-giant-stitch-v0_lidar.npz
    # 2) dump planner reconstructions:
    python -m diffuser.pl_eval.lidar_eval.lidar_planner_recon \
        --planner_logdir logs/antmaze-giant-stitch-v0/diffusion/<lidar-planner-exp> \
        --npz data/ogb_maze/lidar/antmaze-giant-stitch-v0_lidar.npz \
        --out_npz logs/lidar_eval/giant_recon.npz
"""

import sys, os; sys.path.append('./')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('MUJOCO_GL', 'egl')

import json
import argparse
import numpy as np
import torch

import copy

import diffuser.utils as utils
from diffuser.utils import load_config, get_latest_epoch
from diffuser.datasets.normalization import DatasetNormalizer
## identical held-out split as the histogram script, so A and B align
from diffuser.pl_eval.lidar_eval.lidar_pos_histogram import split_episodes


def load_planner_ema(logdir, epoch, device):
    """Load ONLY the planner's EMA diffusion model from a run dir.

    Deliberately avoids ``load_stgl_sml_diffusion`` (which also unpickles the
    trainer config -- pulling in the LiDAR *trainer* module -- and builds a MuJoCo
    renderer, neither of which reconstruction needs). We rebuild the diffusion
    wrapper from model_config.pkl + dfu_model.pkl and load the 'ema' weights from
    state_<epoch>.pt, mirroring what the trainer does (ema_model = deepcopy(model);
    load_state_dict(ckpt['ema']))."""
    unet = load_config(logdir, 'model_config.pkl')()
    dfu_model = load_config(logdir, 'dfu_model.pkl')(model=unet)
    ema_model = copy.deepcopy(dfu_model)
    if epoch == 'latest':
        epoch = get_latest_epoch((logdir,))
    ckpt = torch.load(f'{logdir}/state_{epoch}.pt', map_location=device, weights_only=False)
    ema_model.load_state_dict(ckpt['ema'])
    utils.freeze_model(ema_model)
    return ema_model.to(device).eval(), epoch


def make_lidar_normalizer(lidar, actions):
    """Planner 30-D LiDAR normalizer == LimitsNormalizer fit on all LiDAR scans.
    Computing min/max over the full npz reproduces the training normalizer (same
    scans; first-state padding adds no new extremes)."""
    obs_mm = np.stack([lidar.min(axis=0), lidar.max(axis=0)]).astype(np.float32)
    act_mm = np.stack([actions.min(axis=0), actions.max(axis=0)]).astype(np.float32)
    return DatasetNormalizer(
        {'observations': obs_mm, 'actions': act_mm}, 'LimitsNormalizer', eval_solo=True)


def make_chunks(lidar, xy, test_mask, terminals, horizon, stride, max_chunks, seed):
    """Slide horizon-length windows over the *test* episodes. Returns
    (lidar_chunks [M,H,30], xy_chunks [M,H,2])."""
    n = len(lidar)
    ends = np.where(terminals)[0]
    bounds, prev = [], 0
    for e in ends:
        bounds.append((prev, e + 1)); prev = e + 1
    if prev < n:
        bounds.append((prev, n))

    lid_chunks, xy_chunks = [], []
    for (s, e) in bounds:
        if not test_mask[s]:            # episode is test iff its transitions are
            continue
        last_start = e - horizon
        w = s
        while w <= last_start:
            lid_chunks.append(lidar[w:w + horizon])
            xy_chunks.append(xy[w:w + horizon])
            w += stride
    if len(lid_chunks) == 0:
        raise RuntimeError('no full-horizon chunk in the test split; lower --stride '
                           'or check the episode lengths vs horizon.')
    lid = np.asarray(lid_chunks, dtype=np.float32)
    xyc = np.asarray(xy_chunks, dtype=np.float32)
    if max_chunks > 0 and len(lid) > max_chunks:
        rng = np.random.default_rng(seed)
        sel = rng.permutation(len(lid))[:max_chunks]
        lid, xyc = lid[sel], xyc[sel]
    return lid, xyc


@torch.no_grad()
def reconstruct_batch(model, x_start_t, t_list, n_seeds, device):
    """x_start_t: (b,H,D) normalized clean chunk -> mean predicted x_0 (b,H,D)."""
    b, H, _ = x_start_t.shape
    stgl_cond = {0: x_start_t[:, 0, :], model.horizon - 1: x_start_t[:, model.horizon - 1, :]}
    g_cond = dict(do_cond='both_stgl', stgl_cond=stgl_cond, t_type='0',
                  traj_full=np.zeros(b, dtype=np.float32))
    acc = torch.zeros_like(x_start_t)
    cnt = 0
    for t_scalar in t_list:
        t_2d = torch.full((b, H), int(t_scalar), device=device, dtype=torch.long)
        for _s in range(n_seeds):
            noise = torch.randn_like(x_start_t)
            x_noisy = model.q_sample(x_start_t, t_2d=t_2d, noise=noise)
            ## inpaint the (clean) start/goal scans, build the conditioning dict
            x_cond, tj_cond = model.get_tj_cond(x_noisy, dict(g_cond), t_2d)
            out = model.p_mean_variance(x_cond, t_2d=t_2d, tj_cond=tj_cond, return_modelout=True)
            acc += out[3]                       # x_recon (predicted clean chunk)
            cnt += 1
    return acc / cnt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--planner_logdir', required=True,
                   help='LiDAR planner run dir, e.g. logs/<dset>/diffusion/<exp>')
    p.add_argument('--npz', required=True, help='full LiDAR npz from gen_lidar_dataset.py')
    p.add_argument('--out_npz', required=True, help='output recon npz path')
    p.add_argument('--diffusion_epoch', type=str, default='latest')
    p.add_argument('--cond_w', type=float, default=2.0, help='classifier-free guidance weight')
    p.add_argument('--t_list', type=int, nargs='+', default=None,
                   help='diffusion timesteps to average (default: [T//4])')
    p.add_argument('--n_seeds', type=int, default=2, help='noise draws per timestep')
    p.add_argument('--horizon', type=int, default=None, help='default: model.horizon')
    p.add_argument('--stride', type=int, default=None, help='window stride (default: horizon//4)')
    p.add_argument('--max_chunks', type=int, default=400, help='cap chunks (0 = all)')
    p.add_argument('--batch_size', type=int, default=40)
    p.add_argument('--test_frac', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', type=str, default='cuda:0')
    args = p.parse_args()

    epoch = args.diffusion_epoch if args.diffusion_epoch == 'latest' else int(float(args.diffusion_epoch))

    ## ---- planner EMA model only (normalizer comes from the npz) ----
    model, planner_epoch = load_planner_ema(args.planner_logdir, epoch, args.device)
    model.condition_guidance_w = args.cond_w
    H = args.horizon or model.horizon
    assert H == model.horizon, f'horizon must equal model.horizon ({model.horizon})'
    stride = args.stride or max(1, H // 4)
    t_list = args.t_list or [model.n_timesteps // 4]
    utils.print_color(f'[recon] planner @ epoch {planner_epoch}, obs_dim={model.observation_dim}, '
                      f'H={H}, T={model.n_timesteps}, t_list={t_list}, cond_w={args.cond_w}', c='c')

    ## ---- data + normalizer ----
    d = np.load(args.npz)
    assert 'lidar' in d and 'xy' in d, 'npz must come from gen_lidar_dataset.py (lidar+xy+terminals)'
    lidar = d['lidar'].astype(np.float32)
    xy = d['xy'].astype(np.float32)
    terminals = d['terminals'].astype(bool) if 'terminals' in d else np.zeros(len(lidar), bool)
    actions = d['actions'].astype(np.float32) if 'actions' in d else np.zeros((len(lidar), 1), np.float32)
    assert lidar.shape[1] == model.observation_dim, \
        f'npz n_beams={lidar.shape[1]} != model observation_dim={model.observation_dim}'

    ## sanity: the npz LiDAR geometry should match the planner's training lidar_config
    try:
        pcfg = load_config(args.planner_logdir, 'dataset_config.pkl')
        plc = pcfg._dict['dataset_config']['lidar_config']
        if 'max_range' in d and not np.isclose(float(d['max_range']), float(plc.get('max_range', 12.0))):
            utils.print_color(f'[recon] WARNING npz max_range={float(d["max_range"])} != planner '
                              f'{plc.get("max_range")}; normalizer may be off', c='y')
    except Exception as e:  # noqa: BLE001
        utils.print_color(f'[recon] could not cross-check lidar_config ({e})', c='y')

    norm = make_lidar_normalizer(lidar, actions)

    _train_mask, test_mask, n_ep, n_test = split_episodes(
        terminals, test_frac=args.test_frac, seed=args.seed)
    lid_chunks, xy_chunks = make_chunks(
        lidar, xy, test_mask, terminals, H, stride, args.max_chunks, args.seed)
    utils.print_color(f'[recon] episodes={n_ep} test={n_test} -> {len(lid_chunks)} chunks '
                      f'(H={H}, stride={stride})', c='c')

    ## ---- reconstruct chunk by chunk ----
    sl = slice(1, H - 1)   # drop the two inpainted endpoints
    true_xy_rows, true_lidar_rows, pred_lidar_rows = [], [], []
    for st in range(0, len(lid_chunks), args.batch_size):
        lid_b = lid_chunks[st:st + args.batch_size]      # (b,H,30) world units
        xy_b = xy_chunks[st:st + args.batch_size]        # (b,H,2)
        x_start = torch.as_tensor(
            norm.normalize(lid_b, 'observations'), dtype=torch.float32, device=args.device)
        x_recon = reconstruct_batch(model, x_start, t_list, args.n_seeds, args.device)
        pred = norm.unnormalize(x_recon.detach().cpu().numpy(), 'observations')  # (b,H,30)

        true_xy_rows.append(xy_b[:, sl, :].reshape(-1, 2))
        true_lidar_rows.append(lid_b[:, sl, :].reshape(-1, lid_b.shape[-1]))
        pred_lidar_rows.append(pred[:, sl, :].reshape(-1, pred.shape[-1]))
        utils.print_color(f'[recon] {min(st + args.batch_size, len(lid_chunks))}/{len(lid_chunks)} chunks', c='c')

    true_xy = np.concatenate(true_xy_rows).astype(np.float32)
    true_lidar = np.concatenate(true_lidar_rows).astype(np.float32)
    pred_lidar = np.concatenate(pred_lidar_rows).astype(np.float32)
    rmse = float(np.sqrt(((pred_lidar - true_lidar) ** 2).mean()))

    os.makedirs(os.path.dirname(os.path.abspath(args.out_npz)) or '.', exist_ok=True)
    np.savez_compressed(args.out_npz, true_xy=true_xy, true_lidar=true_lidar,
                        pred_lidar=pred_lidar)
    meta = dict(planner_logdir=args.planner_logdir, epoch=int(planner_epoch),
                n_rows=int(len(true_xy)), horizon=int(H), t_list=list(t_list),
                n_seeds=int(args.n_seeds), cond_w=float(args.cond_w),
                mean_recon_rmse=rmse, out_npz=args.out_npz)
    with open(args.out_npz.replace('.npz', '_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    utils.print_color('\n==================== Option B: planner reconstruction ====================', c='y')
    utils.print_color(f'  rows (transitions) : {len(true_xy)}', c='g')
    utils.print_color(f'  mean LiDAR RMSE    : {rmse:.4f} (world units)', c='g')
    utils.print_color(f'  recon npz -> {args.out_npz}', c='g')
    utils.print_color('  feed it to the histogram script:', c='g')
    utils.print_color(f'    python -m diffuser.pl_eval.lidar_eval.lidar_pos_histogram \\', c='c')
    utils.print_color(f'        --env <env> --npz {args.npz} \\', c='c')
    utils.print_color(f'        --out <out_dir> --recon_npz {args.out_npz}', c='c')
    utils.print_color('==========================================================================\n', c='y')


if __name__ == '__main__':
    main()
