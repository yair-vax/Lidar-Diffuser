#!/usr/bin/env python
"""
Location tracking + failure analysis for the (x, y, theta) LiDAR grid  (PM plan, Stage 3).

Runs grid localization on REAL held-out trajectories and answers two questions:

  * **Location tracking** -- can the model identify its current location from LiDAR?
    We compare *single-scan* localization (K=1, today's setting) against
    *motion-history* localization (K>1, a Markov filter over the scan sequence) and
    report, per history length K, the final-pose error (in cells + degrees) and the
    success rate within a tolerance. Motion history is expected to collapse the
    single-scan ambiguity.

  * **Failure analysis** -- *where* and *why* localization fails. We bin per-cell
    failure rate onto the maze (``failure_map.png``), and correlate each window's error
    with the *distinctiveness* of its true cell (open / symmetric places are inherently
    ambiguous). The breakdown isolates whether failures are structural (low-
    distinctiveness cells) so we know what to improve.

Outputs: ``accuracy_vs_K.png``, ``failure_map.png``, ``example_track.png`` and
``summary.json``. Pure numpy + matplotlib; no torch / MuJoCo.

Example::

    python diffuser/pl_eval/lidar_eval/lidar_localization_eval.py \
        --env antmaze-giant-stitch-v0 \
        --npz ~/.ogbench/data/antmaze-giant-stitch-v0-val.npz \
        --ks 1 2 4 8 --out logs/lidar_eval/loc_eval_giant
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lidar_grid_common as C  # noqa: E402


def ang_err_deg(a, b):
    d = np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)
    return np.degrees(d)


def sample_windows(obs, ep_id, n_windows, K_max, stride, rng):
    """Anchor frames with a full strided in-episode history of length K_max."""
    need = (K_max - 1) * stride
    idx = np.arange(len(obs))
    elig = idx[idx - need >= 0]
    elig = elig[ep_id[elig - need] == ep_id[elig]]
    if len(elig) == 0:
        raise RuntimeError('no eligible windows; lower --ks / --hist_stride')
    return rng.choice(elig, size=min(n_windows, len(elig)), replace=False)


def main():
    ap = C.add_grid_args(argparse.ArgumentParser())
    ap.add_argument('--npz', default='~/.ogbench/data/antmaze-giant-stitch-v0-val.npz')
    ap.add_argument('--ks', type=int, nargs='+', default=[1, 2, 4, 8],
                    help='history lengths (K=1 == single-scan baseline)')
    ap.add_argument('--hist_stride', type=int, default=15,
                    help='dataset frames between consecutive history scans')
    ap.add_argument('--motion_sigma_cells', type=float, default=2.0)
    ap.add_argument('--theta_blur', type=int, default=1)
    ap.add_argument('--n_windows', type=int, default=400)
    ap.add_argument('--success_tol_cells', type=float, default=1.5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='logs/lidar_eval/loc_eval')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    grid = C.LG.load_or_build_grid(args.env, C.lidar_config_from_args(args),
                                   xy_step=args.xy_step, n_theta=args.n_theta,
                                   cache_root=args.cache_root)
    loc = C.LL.GridLocalizer(grid, sigma=args.sigma)
    scanner = grid.scanner
    mu = grid.maze_unit

    obs, ep_id = C.load_obs_npz(args.npz)
    K_max = max(args.ks)
    anchors = sample_windows(obs, ep_id, args.n_windows, K_max, args.hist_stride, rng)
    N = len(anchors)
    print(f'[loc-eval] {N} windows, K={args.ks}, stride={args.hist_stride}, '
          f'success tol {args.success_tol_cells} cells')

    # render the K_max strided history for every window (oldest -> newest = anchor)
    hist_idx = anchors[:, None] - np.arange(K_max)[None, ::-1] * args.hist_stride  # (N,Kmax)
    scans_all = scanner.scan_obs(obs[hist_idx.reshape(-1)], has_quat=True
                                 ).reshape(N, K_max, args.n_beams)
    from_xy, from_yaw = C.LS.obs_to_xy_yaw(obs[anchors], has_quat=True)   # true final pose

    # per-K localization
    perK = {}
    err_cells_byK = {}
    for K in args.ks:
        seq = scans_all[:, K_max - K:, :]              # newest K frames (ends at anchor)
        errs = np.zeros(N); aerr = np.zeros(N)
        for n in range(N):
            r = loc.localize_sequence(seq[n], motion_sigma_cells=args.motion_sigma_cells,
                                      theta_blur=args.theta_blur)
            errs[n] = np.linalg.norm(r['map_xy'][-1] - from_xy[n]) / mu
            aerr[n] = ang_err_deg(r['map_theta'][-1], from_yaw[n])
        succ = errs <= args.success_tol_cells
        perK[str(K)] = dict(
            mean_err_cells=float(errs.mean()), median_err_cells=float(np.median(errs)),
            p90_err_cells=float(np.percentile(errs, 90)),
            mean_theta_err_deg=float(aerr.mean()),
            success_rate=float(succ.mean()))
        err_cells_byK[K] = errs
        print(f'  K={K:2d}: median {np.median(errs):5.2f} cells | mean {errs.mean():5.2f} '
              f'| success {succ.mean()*100:5.1f}% | theta_err {aerr.mean():4.0f} deg')

    # ---- failure analysis at the largest K ------------------------------#
    Kf = max(args.ks)
    errsK = err_cells_byK[Kf]
    fail = errsK > args.success_tol_cells
    # distinctiveness of each window's TRUE cell (small = ambiguous)
    distinct = loc.distinctiveness_map(min_cell_gap=3.0, reduce='mean')
    rc = np.array([grid.xy_to_rc(x, y) for (x, y) in from_xy])
    dvals = distinct[rc[:, 0], rc[:, 1]]
    ok_d = np.isfinite(dvals)
    corr = float(np.corrcoef(dvals[ok_d], errsK[ok_d])[0, 1]) if ok_d.sum() > 2 else float('nan')
    lo = dvals <= np.nanmedian(dvals)
    fa = dict(
        K=Kf, n_fail=int(fail.sum()), fail_rate=float(fail.mean()),
        corr_distinctiveness_vs_error=corr,
        fail_rate_low_distinct=float(fail[ok_d & lo].mean()) if (ok_d & lo).any() else -1.0,
        fail_rate_high_distinct=float(fail[ok_d & ~lo].mean()) if (ok_d & ~lo).any() else -1.0,
    )
    print(f'[loc-eval] failure analysis @K={Kf}: fail {fa["fail_rate"]*100:.1f}%  '
          f"corr(distinct,err)={corr:+.2f}  "
          f"fail@low-distinct {fa['fail_rate_low_distinct']*100:.0f}% vs "
          f"high {fa['fail_rate_high_distinct']*100:.0f}%")

    # per-cell failure-rate map (bin true final cell)
    cells = np.array([grid.cell_of_xy(x, y) for (x, y) in from_xy])
    H, W = scanner.maze_map.shape
    ftot = np.zeros((H, W)); ffail = np.zeros((H, W))
    for (i, j), f in zip(cells, fail):
        ftot[i, j] += 1; ffail[i, j] += f
    frate = np.where(ftot > 0, ffail / np.maximum(ftot, 1), np.nan)
    # map onto the (n_row, n_col) lattice for plotting via the grid convention
    fr_grid = np.full((grid.n_row, grid.n_col), np.nan)
    for rr in range(grid.n_row):
        for cc in range(grid.n_col):
            if grid.free_mask_xy[rr, cc]:
                i, j = grid.cell_of_xy(grid.x_vals[cc], grid.y_vals[rr])
                fr_grid[rr, cc] = frate[i, j]
    C.plot_xy_map(grid, fr_grid, f'{args.out}/failure_map.png',
                  f'Localization failure rate per cell  (K={Kf}, tol {args.success_tol_cells} cells)',
                  cmap='inferno', label='failure rate')

    # accuracy vs K plot
    import matplotlib.pyplot as plt
    ks = args.ks
    med = [perK[str(k)]['median_err_cells'] for k in ks]
    sr = [perK[str(k)]['success_rate'] * 100 for k in ks]
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(ks, med, '-o', color='tab:red', label='median error (cells)')
    ax1.set_xlabel('history length K (scans)'); ax1.set_ylabel('median error (cells)', color='tab:red')
    ax1.set_xticks(ks)
    ax2 = ax1.twinx()
    ax2.plot(ks, sr, '-s', color='tab:blue', label='success rate (%)')
    ax2.set_ylabel('success rate (%)', color='tab:blue'); ax2.set_ylim(0, 100)
    ax1.set_title('Location tracking: single-scan (K=1) vs motion history')
    fig.tight_layout(); fig.savefig(f'{args.out}/accuracy_vs_K.png', dpi=120); plt.close(fig)

    # one example track (true vs estimated) at the largest K
    ex = int(np.argmax(errsK)) if fail.any() else 0
    seq = scans_all[ex, K_max - Kf:, :]
    r = loc.localize_sequence(seq, motion_sigma_cells=args.motion_sigma_cells,
                              theta_blur=args.theta_blur)
    true_win_xy = obs[hist_idx[ex, K_max - Kf:], :2]
    C.plot_trajectory(grid, f'{args.out}/example_track.png',
                      f'Example motion-history track (K={Kf})',
                      true_xy=true_win_xy, est_xy=r['map_xy'])

    summary = dict(env=args.env, n_windows=N, ks=args.ks, hist_stride=args.hist_stride,
                   success_tol_cells=args.success_tol_cells, xy_step=args.xy_step,
                   n_theta=args.n_theta, sigma=args.sigma, per_K=perK, failure_analysis=fa)
    with open(f'{args.out}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'[loc-eval] saved plots + summary.json -> {args.out}')


if __name__ == '__main__':
    main()
