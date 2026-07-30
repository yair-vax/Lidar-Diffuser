"""
Goal-SENSITIVITY probe of the trained LiDAR inv-dyn (no MuJoCo, CPU-ok, ~minutes).

Why this exists (2026-07-05)
----------------------------
The pursuit-v2 gridnav run restored locomotion (replan goal-dists trend down
~2-5 mj/1k steps vs flat in 21129) but ALL SIX ARMS are statistically identical
(rank-sum p=0.37-0.98): BFS routes to the TRUE goal do no better on distance-to-
true-goal than BFS routes to 8-16-cell-off decoys or goal-agnostic real dataset
segments (means all ~47-49 mj). The executed motion is INDEPENDENT of plan
content. Together with June's evidence (every goal-side manipulation changed
nothing; horizon-sweep R^2 ~flat in k), the prime suspect is that the inv-dyn is
(near-)GOAL-BLIND: proprioception alone (gait phase + momentum) predicts the
dataset action to R^2~0.88, so the goal-scan input was gradient-starved at
training time and the model never learned to steer.

What this measures
------------------
On real (state, action, future-scan) pairs at a fixed horizon k (default 4 = the
training sweet spot), compare the model's action under manipulated goals:

    true      goal = the true k-ahead scan                (reference)
    self      goal = the CURRENT scan ("you have arrived") <- the executor's
                                                             hold/stop command
    shuffled  goal = the k-ahead scan of ANOTHER trajectory (wrong place)
    roll2/roll7  goal = true scan circularly rolled 2 / 7 beams (24deg / 84deg
                        heading command; tests whether "turn" is expressible)
    zeros     goal = all-zero vector (degenerate)
    CONTROL: proprio-shuffled, goal true  -> expect MSE to explode (sanity that
             the probe can detect sensitivity at all)

Per condition: action MSE vs dataset actions (+R^2) AND act_delta = median over
samples of ||a(s,g_cond) - a(s,g_true)|| / RMS||a(s,g_true)||.

READ (pre-registered decision rule):
  * shuffled act_delta <~ 0.10 and MSE(shuffled) ~= MSE(true)
        -> GOAL-BLIND CONFIRMED. The executor cannot steer, no eval-time fix can
           work. Fix = make the goal informative to the ACTION at train time
           (wide-k goal sampling + direction-revealing goal encoding, e.g. the
           difference scan g - s_scan), or a different executor (action chunks).
  * shuffled act_delta substantial (>~0.3) but roll2/roll7 deltas tiny/garbage
        -> goal is read but "turn" is not expressible -> same retrain direction.
  * shuffled AND roll deltas substantial
        -> the model CAN steer on REAL pairs; the closed-loop failure is then
           ribbon OOD (tangent-yaw ray-cast scans vs real gait-wobble scans)
           -> build A2: retrieve REAL dataset scans along the BFS route instead
           of synthesizing them (no retrain).

Example
-------
    python -m diffuser.pl_eval.lidar_eval.lidar_invdyn_goal_sensitivity \
        --inv_model_path logs/antmaze-giant-stitch-v0/diffusion/og_antM_Gi_lidar30_full_g30d_invdyn_h12 \
        --inv_epoch latest --device cpu \
        --out logs/lidar_eval/invdyn_goal_sensitivity.json
"""

import sys, os; sys.path.append('./')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('MUJOCO_GL', 'egl')

import json
import argparse
import numpy as np
import torch

import diffuser.utils as utils
from diffuser.ogb_task.og_inv_dyn.og_invdyn_helpers import ogb_load_invdyn_maze_v1
## reuse Option C's helpers so layout + normalizer match training exactly
from diffuser.pl_eval.lidar_eval.lidar_invdyn_mse import build_dataset, get_goal_sel_idxs


@torch.no_grad()
def run_probe(model, dataset, goal_sel, device, k=4, n_batches=100, batch_size=1024,
              seed=0, goal_encoding='abs', act_chunk=1):
    """goal_encoding/act_chunk are auto-read from the run's trainer pkl in main():
    'diff' models get g - current_scan as input (all conditions transformed the
    same way, so 'self' becomes the zero vector -- the trained stop signal);
    act_chunk>1 models are scored on the flattened next-m-action target, with
    per-chunk-index deltas (first/mid/tail) -- the tail is where goal usage is
    forced, so the acceptance reads the TAIL shuffled delta."""
    model = model.to(device).eval()
    goal_sel_t = torch.as_tensor(list(goal_sel), dtype=torch.long)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                         num_workers=4, pin_memory=False)
    rng = np.random.RandomState(seed)

    conds = ['true', 'self', 'shuffled', 'roll2', 'roll7', 'zeros', 'proprio_shuf']
    sums = {c: dict(se=None, n=0) for c in conds}
    deltas = {c: [] for c in conds}          # ||a_cond - a_true|| per sample
    m = max(int(act_chunk), 1)
    tail_js = sorted(set([0, m // 2, m - 1])) if m > 1 else [0]
    deltas_j = {c: {j: [] for j in tail_js} for c in conds}   # per-chunk-index
    a_true_norms = []
    sum_a = None; sum_a2 = None; n_a = 0

    for bi, batch in enumerate(loader):
        if n_batches is not None and bi >= n_batches:
            break
        obs_trajs, act_trajs, _cond, val_lens = batch
        obs_trajs = obs_trajs.float(); act_trajs = act_trajs.float()
        b_s = obs_trajs.shape[0]
        if b_s < 4:
            continue
        b_idxs = np.arange(b_s)
        st_idxs = np.zeros(b_s, dtype=np.int64)
        val_np = np.asarray(val_lens).reshape(-1).astype(np.int64)
        goal_idxs = np.minimum(st_idxs + k, val_np - 1)
        valid = goal_idxs > st_idxs

        x_t = obs_trajs[b_idxs, st_idxs, :].to(device)                       # (B, obs_dim)
        if m > 1:
            assert act_trajs.shape[1] >= m, (act_trajs.shape, m)
            a_t = act_trajs[:, :m, :].reshape(b_s, -1).to(device)            # (B, m*8)
        else:
            a_t = act_trajs[b_idxs, st_idxs, :].to(device)                   # (B, 8)
        g_true = obs_trajs[b_idxs, goal_idxs, :][:, goal_sel_t].to(device)   # (B, nb)
        g_self = x_t[:, goal_sel_t]                                          # current scan
        perm = rng.permutation(b_s)
        ## make sure "shuffled" really is another trajectory
        clash = perm == b_idxs
        if clash.any():
            perm[clash] = (perm[clash] + 1) % b_s
        g_shuf = g_true[perm]
        g_roll2 = torch.roll(g_true, shifts=2, dims=1)
        g_roll7 = torch.roll(g_true, shifts=7, dims=1)
        g_zero = torch.zeros_like(g_true)

        def enc(g):
            ## difference-goal models see g - current scan (zeros stays zeros
            ## only for the 'self' condition, by construction)
            return (g - x_t[:, goal_sel_t]) if goal_encoding == 'diff' else g

        vmask = torch.as_tensor(valid, device=device).float().unsqueeze(1)
        a_true_pred = model(x_t, enc(g_true))

        goal_map = dict(true=g_true, self=g_self, shuffled=g_shuf,
                        roll2=g_roll2, roll7=g_roll7, zeros=g_zero)
        for c in conds:
            if c == 'proprio_shuf':
                xp = x_t[perm]
                gp = (g_true - xp[:, goal_sel_t]) if goal_encoding == 'diff' else g_true
                pred = model(xp, gp)                 # wrong body state, right goal
            else:
                pred = model(x_t, enc(goal_map[c]))
            se = (((pred - a_t) ** 2) * vmask).sum(dim=0)
            sums[c]['se'] = se if sums[c]['se'] is None else sums[c]['se'] + se
            sums[c]['n'] += int(valid.sum())
            d = torch.linalg.norm(pred - a_true_pred, dim=1)[torch.as_tensor(valid)]
            deltas[c].append(d.cpu().numpy())
            if m > 1:
                pj = pred.reshape(b_s, m, -1); tj = a_true_pred.reshape(b_s, m, -1)
                for j in tail_js:
                    dj = torch.linalg.norm(pj[:, j, :] - tj[:, j, :], dim=1)
                    deltas_j[c][j].append(dj[torch.as_tensor(valid)].cpu().numpy())

        a_true_norms.append(torch.linalg.norm(a_true_pred, dim=1)[torch.as_tensor(valid)].cpu().numpy())
        a_masked = a_t * vmask
        sum_a = a_masked.sum(0) if sum_a is None else sum_a + a_masked.sum(0)
        sum_a2 = (a_masked * a_t).sum(0) if sum_a2 is None else sum_a2 + (a_masked * a_t).sum(0)
        n_a += int(valid.sum())

    rms_a = float(np.sqrt(np.mean(np.concatenate(a_true_norms) ** 2)))
    mean_a = (sum_a / max(n_a, 1)).cpu().numpy()
    var_a = float(((sum_a2 / max(n_a, 1)).cpu().numpy() - mean_a ** 2).mean())

    out = {'k': k, 'n_transitions': n_a, 'predict_mean_mse(=var)': var_a,
           'rms_pred_action_norm': rms_a, 'goal_encoding': goal_encoding,
           'act_chunk': m, 'conditions': {}}
    ## per-index normalizer: rms of the true-goal prediction at that index scales
    ## with rms_a/sqrt(m) per index (norm over act_dim vs m*act_dim), so ratios
    ## use rms_a/sqrt(m) -- comparable to the single-action probe's scale.
    rms_j = rms_a / np.sqrt(m)
    for c in conds:
        n = max(sums[c]['n'], 1)
        mse = float((sums[c]['se'] / n).mean())
        dd = np.concatenate(deltas[c]) if deltas[c] else np.array([0.0])
        entry = {
            'action_mse': mse,
            'r2': float(1.0 - mse / var_a) if var_a > 1e-9 else float('nan'),
            'act_delta_median': float(np.median(dd)),
            'act_delta_ratio': float(np.median(dd) / max(rms_a, 1e-9)),
        }
        if m > 1:
            entry['chunk_delta_ratio_by_index'] = {
                str(j): float(np.median(np.concatenate(deltas_j[c][j])) / max(rms_j, 1e-9))
                for j in tail_js if deltas_j[c][j]}
        out['conditions'][c] = entry
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inv_model_path', type=str,
                   default='logs/antmaze-giant-stitch-v0/diffusion/og_antM_Gi_lidar30_full_g30d_invdyn_h12')
    p.add_argument('--inv_epoch', type=str, default='latest')
    p.add_argument('--dset_h5path', type=str, default=None,
                   help='None = train split; pass the val npz for held-out')
    p.add_argument('--k', type=int, default=4)
    p.add_argument('--n_batches', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=1024)
    p.add_argument('--use_model', action='store_true')
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=str, default='logs/lidar_eval/invdyn_goal_sensitivity.json')
    args = p.parse_args()

    inv_epoch = args.inv_epoch if args.inv_epoch == 'latest' else int(float(args.inv_epoch))
    model, ema_model, epoch = ogb_load_invdyn_maze_v1(args.inv_model_path, epoch=inv_epoch)
    eval_model = model if args.use_model else ema_model
    dataset = build_dataset(args.inv_model_path, dset_h5path=args.dset_h5path)
    goal_sel = get_goal_sel_idxs(args.inv_model_path, dataset)

    ## v2 (2026-07-07): auto-detect encoding + chunk from the RUN'S OWN trainer
    ## pkl -- a diff/chunk model probed with absolute single-action assumptions
    ## would read as spuriously blind (the widek-vs-h12 default-path lesson).
    goal_encoding, act_chunk = 'abs', 1
    try:
        from diffuser.utils import load_config
        _td = load_config(args.inv_model_path, 'inv_trainer_config.pkl') \
            ._dict.get('trainer_dict', {}) or {}
        goal_encoding = str(_td.get('goal_encoding', 'abs'))
        act_chunk = int(_td.get('act_chunk', 1) or 1)
    except Exception as e:  # noqa: BLE001
        utils.print_color(f'[probe] no trainer pkl ({e}); assuming abs/1', c='y')
    if act_chunk > 1 and args.k == 4:
        utils.print_color(f'[probe] NOTE: chunk model (m={act_chunk}) probed at the '
                          f'default k=4; consider --k {act_chunk} (the trained mid)', c='y')

    res = run_probe(eval_model, dataset, goal_sel, args.device, k=args.k,
                    n_batches=(None if args.n_batches < 0 else args.n_batches),
                    batch_size=args.batch_size, seed=args.seed,
                    goal_encoding=goal_encoding, act_chunk=act_chunk)
    res.update(inv_model_path=args.inv_model_path, inv_epoch=epoch,
               model_kind='model' if args.use_model else 'ema',
               split='train' if args.dset_h5path is None else args.dset_h5path)

    utils.print_color('\n===== inv-dyn GOAL SENSITIVITY (k=%d, %d trans, enc=%s, chunk=%d) =====' %
                      (res['k'], res['n_transitions'], res['goal_encoding'], res['act_chunk']), c='y')
    utils.print_color('  condition      MSE      R^2    act_delta_ratio (vs true-goal action)', c='g')
    for c, r in res['conditions'].items():
        line = f"  {c:12s} {r['action_mse']:.5f}  {r['r2']:.3f}   {r['act_delta_ratio']:.3f}"
        if 'chunk_delta_ratio_by_index' in r:
            prof = '  '.join(f'j{j}:{v:.3f}' for j, v in r['chunk_delta_ratio_by_index'].items())
            line += f'   [{prof}]'
        utils.print_color(line, c='g')
    if res['act_chunk'] > 1:
        utils.print_color('  CHUNK READ: acceptance = shuffled TAIL (last j) ratio >= ~0.4 and the', c='y')
        utils.print_color('  profile RISING (first < mid < tail) -> the loss forced goal usage.', c='y')
    utils.print_color('  predict-mean var = %.4f, RMS pred-action = %.3f' %
                      (res['predict_mean_mse(=var)'], res['rms_pred_action_norm']), c='c')
    utils.print_color('  READ: shuffled delta <~0.10 & MSE(shuffled)~=MSE(true) -> GOAL-BLIND ->', c='y')
    utils.print_color('        retrain inv-dyn goal-aware. Deltas substantial -> model steers on', c='y')
    utils.print_color('        real pairs -> closed-loop failure is ribbon OOD -> build A2', c='y')
    utils.print_color('        (real-dataset scans along the BFS route). proprio_shuf must be BIG', c='y')
    utils.print_color('        (sanity); self-vs-true delta = the executor\'s stop-vs-go signal.\n', c='y')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=2)
    utils.print_color(f'  saved -> {args.out}\n', c='c')


if __name__ == '__main__':
    main()
