"""
Teacher-forcing diagnostic (no MuJoCo) -- inv-dyn action MSE vs GOAL HORIZON.

Why this exists
---------------
Option C (``lidar_invdyn_mse.py``) showed action MSE ~= 0.335 on the *train* split,
which is ~ the variance of the actions => the inv-dyn barely beats predicting the mean
(R^2 ~ 0). The hypothesis (see LIDAR_DIAGNOSIS_AND_BASELINE.md) is that the goal is
*ill-posed*: the trainer asks for the FIRST action given a LiDAR scan sampled at a
RANDOM future step in [1, horizon). The same (state, goal-scan) then maps to many valid
first actions, so the L2-optimal answer collapses to the mean.

This script is the cheap, decisive form of the "open-loop teacher-forcing" test: it
feeds the inv-dyn the *true* future scan at a CONTROLLED horizon k and measures whether
the predicted action matches the true action. If MSE at small k (well-posed, near goal)
is much lower than at the random baseline, the formulation -- not the network -- is the
problem, and the fix is a fixed small-horizon inverse dynamics (see
``og_invdyn_training_reposed.py`` + the ``..._fix1.py`` config).

It also reports the predict-the-mean variance baseline so MSE is interpretable as R^2:
    R^2 = 1 - MSE / Var(action).

No MuJoCo / rendering. Needs GPU + the inv-dyn checkpoint + the dataset + LiDAR cache.

Example
-------
    python -m diffuser.pl_eval.lidar_eval.lidar_invdyn_horizon_sweep \
        --inv_model_path logs/antmaze-giant-stitch-v0/diffusion/og_antM_Gi_lidar30_xyo_g30d_invdyn_h12 \
        --inv_epoch 800000 \
        --out logs/lidar_eval/invdyn_horizon_sweep.json
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
## reuse Option C's dataset/goal-idx helpers so layout + normalizer match training exactly
from diffuser.pl_eval.lidar_eval.lidar_invdyn_mse import build_dataset, get_goal_sel_idxs


@torch.no_grad()
def eval_mse_at_horizon(model, dataset, goal_sel, device, n_batches, batch_size,
                        mode='fixed', k=1, n_goal_reps=1, seed=0):
    """Action MSE when the goal scan is taken at a controlled horizon.

    mode='fixed': goal index = clip(0 + k, 1, val_len-1)         (well-posed, k steps ahead)
    mode='rand' : goal index = randint(1, hzn) clip val_len-1     (the trainer's own scheme)

    Returns dict with overall/per-dim action MSE, the predict-mean variance baseline,
    R^2, and the count.
    """
    model = model.to(device).eval()
    goal_sel = torch.as_tensor(list(goal_sel), dtype=torch.long)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                         num_workers=4, pin_memory=False)
    rng = np.random.RandomState(seed)

    sum_se = None          # running sum of squared errors per action dim
    sum_a = None           # running sum of actions (for the mean baseline)
    sum_a2 = None          # running sum of action^2
    n_count = 0
    for bi, batch in enumerate(loader):
        if n_batches is not None and bi >= n_batches:
            break
        obs_trajs, act_trajs, _cond, val_lens = batch
        obs_trajs = obs_trajs.float(); act_trajs = act_trajs.float()
        b_s, hzn = obs_trajs.shape[0], obs_trajs.shape[1]
        b_idxs = np.arange(b_s)
        st_idxs = np.zeros(b_s, dtype=np.int64)
        val_np = np.asarray(val_lens).reshape(-1).astype(np.int64)

        x_t = obs_trajs[b_idxs, st_idxs, :].to(device)        # (B, 34)
        a_t = act_trajs[b_idxs, st_idxs, :].to(device)        # (B, 8)

        for _rep in range(n_goal_reps):
            if mode == 'fixed':
                goal_idxs = np.minimum(st_idxs + k, val_np - 1)   # k steps ahead, clipped
            elif mode == 'rand':
                goal_idxs = rng.randint(low=st_idxs + 1, high=hzn)
                goal_idxs = np.clip(goal_idxs, 0, val_np - 1)
            else:
                raise ValueError(mode)
            ## guard: when val_len==1 there is no valid future step; skip those rows
            valid = goal_idxs > st_idxs
            x_t_1 = obs_trajs[b_idxs, goal_idxs, :][:, goal_sel].to(device)   # (B, 30)
            pred = model(x_t, x_t_1)                                          # (B, 8)
            vmask = torch.as_tensor(valid, device=device).float().unsqueeze(1)
            se = (((pred - a_t) ** 2) * vmask).sum(dim=0)
            sum_se = se if sum_se is None else sum_se + se
            a_masked = a_t * vmask
            sum_a = a_masked.sum(0) if sum_a is None else sum_a + a_masked.sum(0)
            sum_a2 = (a_masked * a_t).sum(0) if sum_a2 is None else sum_a2 + (a_masked * a_t).sum(0)
            n_count += int(valid.sum())

    n = max(n_count, 1)
    per_dim_mse = (sum_se / n).detach().cpu().numpy()
    mean_a = (sum_a / n).detach().cpu().numpy()
    var_a = (sum_a2 / n).detach().cpu().numpy() - mean_a ** 2     # predict-mean MSE per dim
    overall_mse = float(per_dim_mse.mean())
    overall_var = float(var_a.mean())
    r2 = float(1.0 - overall_mse / overall_var) if overall_var > 1e-9 else float('nan')
    return {
        'action_mse': overall_mse,
        'per_dim_action_mse': [float(v) for v in per_dim_mse],
        'predict_mean_mse(=var)': overall_var,
        'r2': r2,
        'n_transitions': int(n_count),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inv_model_path', type=str,
                   default='logs/antmaze-giant-stitch-v0/diffusion/og_antM_Gi_lidar30_xyo_g30d_invdyn_h12')
    p.add_argument('--inv_epoch', type=str, default='800000')
    p.add_argument('--dset_h5path', type=str, default=None)
    p.add_argument('--ks', type=int, nargs='+', default=[1, 2, 3, 4, 6, 8, 11],
                   help='fixed goal horizons to sweep')
    p.add_argument('--n_batches', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=1024)
    p.add_argument('--use_model', action='store_true', help='score raw model not EMA')
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=str, default='logs/lidar_eval/invdyn_horizon_sweep.json')
    args = p.parse_args()

    inv_epoch = args.inv_epoch if args.inv_epoch == 'latest' else int(float(args.inv_epoch))
    n_batches = None if args.n_batches < 0 else args.n_batches

    model, ema_model, epoch = ogb_load_invdyn_maze_v1(args.inv_model_path, epoch=inv_epoch)
    eval_model = model if args.use_model else ema_model
    dataset = build_dataset(args.inv_model_path, dset_h5path=args.dset_h5path)
    goal_sel = get_goal_sel_idxs(args.inv_model_path, dataset)

    results = {'inv_model_path': args.inv_model_path, 'inv_epoch': epoch,
               'model_kind': 'model' if args.use_model else 'ema',
               'split': 'train' if args.dset_h5path is None else args.dset_h5path,
               'fixed_horizon': {}, 'random_baseline': None}

    utils.print_color('\n=========== inv-dyn action MSE vs goal horizon ===========', c='y')
    utils.print_color('  k        MSE      var(=predict-mean)   R^2', c='g')
    for k in args.ks:
        r = eval_mse_at_horizon(eval_model, dataset, goal_sel, args.device,
                                n_batches, args.batch_size, mode='fixed', k=k, seed=args.seed)
        results['fixed_horizon'][str(k)] = r
        utils.print_color(f'  k={k:<3d}   {r["action_mse"]:.5f}   {r["predict_mean_mse(=var)"]:.5f}"'
                          f'           {r["r2"]:.3f}', c='g')
    rb = eval_mse_at_horizon(eval_model, dataset, goal_sel, args.device,
                             n_batches, args.batch_size, mode='rand', n_goal_reps=4, seed=args.seed)
    results['random_baseline'] = rb
    utils.print_color(f'  rand    {rb["action_mse"]:.5f}   {rb["predict_mean_mse(=var)"]:.5f}'
                      f'           {rb["r2"]:.3f}   <- the trainer\'s current scheme', c='c')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    utils.print_color(f'\n  saved -> {args.out}', c='c')
    utils.print_color('  READ: if MSE at k=1 is much lower (R^2 >> 0) than the random baseline,', c='y')
    utils.print_color('        the goal formulation is the bug -> retrain with the fix-1 config.\n', c='y')


if __name__ == '__main__':
    main()
