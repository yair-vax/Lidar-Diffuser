"""
Option C -- inverse-dynamics action-MSE check for the LiDAR antmaze model.

Question it answers: *did the LiDAR inverse-dynamics model actually learn?* It is
the only thing standing between a generated LiDAR plan and motor commands, so verify
it before trusting it in the closed-loop rollout (Option D).

What it does (mirrors the trainer's own sampling, see
``diffuser/ogb_task/og_inv_dyn/og_invdyn_training.py`` train loop):

    x_t   = obs_trajs[:, 0, :]                 # (B, 34) = [x, y, cos, sin, lidar]
    g_idx = random valid future index in [1, val_len-1]
    goal  = obs_trajs[:, g_idx, goal_sel_idxs] # (B, 30) the LiDAR slice [4:34]
    a_t   = act_trajs[:, 0, :]                 # (B, 8)
    pred  = inv_model(x_t, goal)               # MLP_InvDyn_OgB_V3.forward (means)
    mse   = mean( (pred - a_t)^2 )             # normalized action space

It reports the overall MSE (same units / normalized action space as the wandb train
loss -- compare them), the per-action-dim MSE, and -- if a held-out split is given
via ``--val_h5path`` -- the train-vs-held-out gap (an overfit check).

By default it evaluates on the **training** dataset (the data the inv-dyn was trained
on), which is the cheapest "did it learn?" sanity check: this number should match the
final wandb train loss. For a genuine generalization / overfit check, pass
``--val_h5path <ogbench val npz>``; the held-out observations are then normalized with
the *training* normalizer (so the model sees in-distribution inputs) before scoring.

No MuJoCo rendering needed (no rollout). Run on the server (needs GPU + the LiDAR
inv-dyn checkpoint + the dataset + the disk LiDAR cache).

Example::

    python -m diffuser.pl_eval.lidar_eval.lidar_invdyn_mse \
        --inv_model_path logs/antmaze-giant-stitch-v0/diffusion/og_antM_Gi_lidar30_xyo_g30d_invdyn_h12 \
        --inv_epoch 800000 \
        --out logs/lidar_eval/invdyn_mse.json
"""

import sys, os; sys.path.append('./')
## EGL just in case env creation touches GL; no rendering is done here.
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('MUJOCO_GL', 'egl')

import json
import argparse
import numpy as np
import torch

import diffuser.utils as utils
from diffuser.utils import load_config
from diffuser.ogb_task.og_inv_dyn.og_invdyn_helpers import ogb_load_invdyn_maze_v1


def build_dataset(inv_model_path, dset_h5path=None):
    """Rebuild the inv-dyn dataset from the run's saved dataset_config.pkl (so the
    LiDAR layout / normalizer match training exactly). Optionally swap the data
    source (e.g. to the OGBench validation npz)."""
    cfg = load_config(inv_model_path, 'dataset_config.pkl')
    if dset_h5path is not None:
        cfg._dict['dset_h5path'] = dset_h5path
    return cfg()


def get_goal_sel_idxs(inv_model_path, dataset):
    """The dims of the obs the model treats as 'goal'. Authoritative source is the
    saved trainer config; fall back to the LiDAR layout (the [4:] LiDAR slice)."""
    try:
        tr_cfg = load_config(inv_model_path, 'inv_trainer_config.pkl')
        return list(tr_cfg._dict['goal_sel_idxs'])
    except Exception as e:  # noqa: BLE001
        utils.print_color(f'[invdyn_mse] no trainer config ({e}); deriving from lidar_config', c='y')
        from diffuser.datasets.ogb_dset.ogb_lidar_utils import lidar_goal_idxs
        return list(lidar_goal_idxs(dataset.dataset_config['lidar_config']))


@torch.no_grad()
def eval_action_mse(model, dataset, goal_sel_idxs, device='cuda:0',
                    n_batches=None, batch_size=1024, n_goal_reps=4, seed=0):
    """Compute the normalized-action MSE mirroring the trainer's sampling.
    Returns (overall_mse, per_dim_mse [act_dim], n_transitions)."""
    model = model.to(device).eval()
    ## index on CPU (numpy advanced indexing, exactly like the trainer), then move
    ## only the selected slices to the GPU -- avoids any cpu/cuda index mismatch.
    goal_sel = torch.as_tensor(list(goal_sel_idxs), dtype=torch.long)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=False)
    rng = np.random.RandomState(seed)

    sum_se = None   # (act_dim,) running sum of squared errors
    n_count = 0
    for bi, batch in enumerate(loader):
        if n_batches is not None and bi >= n_batches:
            break
        obs_trajs, act_trajs, _conditions, val_lens = batch
        obs_trajs = obs_trajs.float()                  # keep on CPU for indexing
        act_trajs = act_trajs.float()

        b_s, hzn = obs_trajs.shape[0], obs_trajs.shape[1]
        b_idxs = np.arange(b_s)
        st_idxs = np.zeros(b_s, dtype=np.int64)
        val_lens_np = np.asarray(val_lens).reshape(-1).astype(np.int64)

        ## x_t / a_t are the start state and the first action (st_idx == 0)
        x_t = obs_trajs[b_idxs, st_idxs, :].to(device)            # (B, obs_dim)
        a_t = act_trajs[b_idxs, st_idxs, :].to(device)            # (B, act_dim)

        for _rep in range(n_goal_reps):
            ## random valid future index, exactly as the trainer does
            goal_idxs = rng.randint(low=st_idxs + 1, high=hzn)   # (B,)
            goal_idxs = np.clip(goal_idxs, a_min=0, a_max=val_lens_np - 1)
            x_t_1 = obs_trajs[b_idxs, goal_idxs, :][:, goal_sel].to(device)  # (B, goal_dim)

            pred = model(x_t, x_t_1)                    # (B, act_dim) means
            se = ((pred - a_t) ** 2).sum(dim=0)         # (act_dim,)
            sum_se = se if sum_se is None else sum_se + se
            n_count += b_s

    per_dim_mse = (sum_se / max(n_count, 1)).detach().cpu().numpy()
    overall_mse = float(per_dim_mse.mean())
    return overall_mse, per_dim_mse, int(n_count)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inv_model_path', type=str,
                   default='logs/antmaze-giant-stitch-v0/diffusion/og_antM_Gi_lidar30_xyo_g30d_invdyn_h12')
    p.add_argument('--inv_epoch', type=str, default='800000',
                   help="saved inv-dyn step, or 'latest'")
    p.add_argument('--dset_h5path', type=str, default=None,
                   help='override the dataset source for the *main* eval (default: training set)')
    p.add_argument('--val_h5path', type=str, default=None,
                   help='OGBench validation npz -> genuine held-out MSE + train gap')
    p.add_argument('--n_batches', type=int, default=200,
                   help='number of batches to score (-1 = whole dataset)')
    p.add_argument('--batch_size', type=int, default=1024)
    p.add_argument('--n_goal_reps', type=int, default=4,
                   help='avg over this many random goal-index draws (variance reduction)')
    p.add_argument('--use_model', action='store_true',
                   help='score the raw model instead of the EMA model (default: EMA)')
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=str, default='logs/lidar_eval/invdyn_mse.json')
    args = p.parse_args()

    inv_epoch = args.inv_epoch if args.inv_epoch == 'latest' else int(float(args.inv_epoch))
    n_batches = None if args.n_batches < 0 else args.n_batches

    ## ---- load the trained inv-dyn (model + ema) ----
    model, ema_model, epoch = ogb_load_invdyn_maze_v1(args.inv_model_path, epoch=inv_epoch)
    eval_model = model if args.use_model else ema_model
    utils.print_color(
        f'[invdyn_mse] loaded {("model" if args.use_model else "ema")} @ epoch {epoch} '
        f'from {args.inv_model_path}', c='c')

    ## ---- main dataset (training set unless --dset_h5path) ----
    train_ds = build_dataset(args.inv_model_path, dset_h5path=args.dset_h5path)
    goal_sel_idxs = get_goal_sel_idxs(args.inv_model_path, train_ds)
    utils.print_color(f'[invdyn_mse] goal_sel_idxs={goal_sel_idxs[:3]}..{goal_sel_idxs[-1]} '
                      f'(n={len(goal_sel_idxs)})', c='c')

    main_split = 'train' if args.dset_h5path is None else f'dset:{args.dset_h5path}'
    mse, per_dim, n_tr = eval_action_mse(
        eval_model, train_ds, goal_sel_idxs, device=args.device,
        n_batches=n_batches, batch_size=args.batch_size,
        n_goal_reps=args.n_goal_reps, seed=args.seed)

    result = {
        'inv_model_path': args.inv_model_path,
        'inv_epoch': epoch,
        'model_kind': 'model' if args.use_model else 'ema',
        'main_split': main_split,
        'n_transitions_scored': n_tr,
        'action_mse': mse,
        'per_dim_action_mse': [float(v) for v in per_dim],
    }

    ## ---- optional genuine held-out (val) split ----
    if args.val_h5path is not None:
        utils.print_color(f'[invdyn_mse] building held-out split from {args.val_h5path}', c='c')
        val_ds = build_dataset(args.inv_model_path, dset_h5path=args.val_h5path)
        ## normalize the held-out data with the TRAINING normalizer so the model
        ## sees in-distribution inputs (not the val split's own min/max).
        val_ds.normalizer = train_ds.normalizer
        val_ds.normalize()
        val_mse, val_per_dim, n_va = eval_action_mse(
            eval_model, val_ds, goal_sel_idxs, device=args.device,
            n_batches=n_batches, batch_size=args.batch_size,
            n_goal_reps=args.n_goal_reps, seed=args.seed)
        result.update({
            'heldout_action_mse': val_mse,
            'heldout_per_dim_action_mse': [float(v) for v in val_per_dim],
            'heldout_n_transitions': n_va,
            'train_minus_heldout_gap': mse - val_mse,
        })

    ## ---- report ----
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2)

    utils.print_color('\n==================== Option C: inv-dyn action MSE ====================', c='y')
    utils.print_color(f'  split           : {main_split}', c='g')
    utils.print_color(f'  action MSE      : {mse:.6f}   (compare to the final wandb train loss)', c='g')
    utils.print_color(f'  per-dim MSE     : {np.round(per_dim, 5).tolist()}', c='g')
    utils.print_color(f'  n transitions   : {n_tr}', c='g')
    if args.val_h5path is not None:
        utils.print_color(f'  held-out MSE    : {result["heldout_action_mse"]:.6f}', c='g')
        utils.print_color(f'  train-heldout   : {result["train_minus_heldout_gap"]:.6f}  '
                          f'(large negative gap => overfit)', c='g')
    utils.print_color(f'  saved -> {args.out}', c='c')
    utils.print_color('=====================================================================\n', c='y')


if __name__ == '__main__':
    main()
