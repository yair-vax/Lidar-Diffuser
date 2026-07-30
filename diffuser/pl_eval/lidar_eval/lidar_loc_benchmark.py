#!/usr/bin/env python
"""
100-run localization benchmark (CPU)  --  PM plan, Stage 3 "Scenario Benchmarking".

Defines a **standard scenario** and runs it many times to measure whether grid
localization *consistently* succeeds. Two blocks are reported:

  A. **Consistency** -- ONE fixed standard scenario (a seeded real trajectory),
     localized ``--n_runs`` (default 100) times, each with independent simulated
     LiDAR sensor NOISE. Answers "run it 100 times consecutively: does it reliably
     succeed?" -> success rate + error mean/std + a histogram.

  B. **Coverage** -- ``--n_scenarios`` (default 100) DISTINCT random start/goal
     windows across the maze, localized once each. Answers "does it work everywhere,
     or only in easy spots?" -> map-wide success rate + a failure breakdown by cell
     distinctiveness.

Both use motion-history localization (the Markov filter); set ``--K 1`` to benchmark
the single-scan baseline instead. Pure numpy + matplotlib; the CPU precursor to the
closed-loop (GPU) 100-run navigation benchmark.

Example::

    python diffuser/pl_eval/lidar_eval/lidar_loc_benchmark.py \
        --env antmaze-giant-stitch-v0 \
        --npz ~/.ogbench/data/antmaze-giant-stitch-v0-val.npz \
        --n_runs 100 --n_scenarios 100 --out logs/lidar_eval/loc_bench_giant
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lidar_grid_common as C  # noqa: E402


def eligible_anchors(obs, ep_id, K, stride):
    need = (K - 1) * stride
    idx = np.arange(len(obs))
    elig = idx[idx - need >= 0]
    return elig[ep_id[elig - need] == ep_id[elig]]


def render_history(scanner, obs, anchor, K, stride, n_beams):
    """(K, n_beams) strided history ending at ``anchor`` (oldest -> newest)."""
    idx = anchor - np.arange(K)[::-1] * stride
    return scanner.scan_obs(obs[idx], has_quat=True), obs[idx, :2]


def main():
    ap = C.add_grid_args(argparse.ArgumentParser())
    ap.add_argument('--npz', default='~/.ogbench/data/antmaze-giant-stitch-v0-val.npz')
    ap.add_argument('--n_runs', type=int, default=100, help='block A: repeats of the standard scenario')
    ap.add_argument('--n_scenarios', type=int, default=100, help='block B: distinct scenarios')
    ap.add_argument('--K', type=int, default=8, help='history length (1 == single-scan baseline)')
    ap.add_argument('--hist_stride', type=int, default=15)
    ap.add_argument('--motion_sigma_cells', type=float, default=2.0)
    ap.add_argument('--noise_std', type=float, default=0.15,
                    help='block A: per-beam LiDAR noise std (world units)')
    ap.add_argument('--success_tol_cells', type=float, default=1.5)
    ap.add_argument('--scenario_seed', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='logs/lidar_eval/loc_bench')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    grid = C.LG.load_or_build_grid(args.env, C.lidar_config_from_args(args),
                                   xy_step=args.xy_step, n_theta=args.n_theta,
                                   cache_root=args.cache_root)
    loc = C.LL.GridLocalizer(grid, sigma=args.sigma)
    scanner = grid.scanner
    mu = grid.maze_unit
    obs, ep_id = C.load_obs_npz(args.npz)
    elig = eligible_anchors(obs, ep_id, args.K, args.hist_stride)
    if len(elig) == 0:
        raise RuntimeError('no eligible anchors; lower --K / --hist_stride')

    def localize_err(scans, true_final_xy):
        r = loc.localize_sequence(scans, motion_sigma_cells=args.motion_sigma_cells,
                                  theta_blur=1)
        return np.linalg.norm(r['map_xy'][-1] - true_final_xy) / mu

    # ---- Block A: standard scenario x n_runs (sensor noise) -------------#
    srng = np.random.default_rng(args.scenario_seed)
    anchor = int(srng.choice(elig))
    base_scans, win_xy = render_history(scanner, obs, anchor, args.K,
                                        args.hist_stride, args.n_beams)
    true_final = win_xy[-1]
    nrng = np.random.default_rng(args.seed)
    errsA = np.zeros(args.n_runs)
    for i in range(args.n_runs):
        noisy = base_scans + nrng.normal(0, args.noise_std, size=base_scans.shape)
        noisy = np.clip(noisy, 0.0, scanner.max_range).astype(np.float32)
        errsA[i] = localize_err(noisy, true_final)
    succA = errsA <= args.success_tol_cells
    blockA = dict(scenario_anchor=int(anchor),
                  scenario_cell=list(grid.cell_of_xy(*true_final)),
                  n_runs=args.n_runs, noise_std=args.noise_std,
                  success_rate=float(succA.mean()),
                  mean_err_cells=float(errsA.mean()), std_err_cells=float(errsA.std()),
                  median_err_cells=float(np.median(errsA)),
                  max_err_cells=float(errsA.max()))
    print(f'[bench A] standard scenario cell={blockA["scenario_cell"]} x {args.n_runs} runs '
          f'(noise {args.noise_std}): SUCCESS {succA.mean()*100:.0f}/100  '
          f'err {errsA.mean():.2f}+/-{errsA.std():.2f} cells')

    # ---- Block B: distinct scenarios (coverage) -------------------------#
    brng = np.random.default_rng(args.seed + 1)
    anchors = brng.choice(elig, size=min(args.n_scenarios, len(elig)), replace=False)
    errsB = np.zeros(len(anchors)); cellsB = []
    for k, a in enumerate(anchors):
        sc, wxy = render_history(scanner, obs, int(a), args.K, args.hist_stride, args.n_beams)
        errsB[k] = localize_err(sc.astype(np.float32), wxy[-1])
        cellsB.append(grid.cell_of_xy(*wxy[-1]))
    succB = errsB <= args.success_tol_cells
    # failure breakdown by cell distinctiveness
    distinct = loc.distinctiveness_map(min_cell_gap=3.0)
    rc = np.array([grid.xy_to_rc(*obs[a, :2]) for a in anchors])
    dv = distinct[rc[:, 0], rc[:, 1]]; ok = np.isfinite(dv)
    lo = ok & (dv <= np.nanmedian(dv[ok]))
    hi = ok & (dv > np.nanmedian(dv[ok]))
    blockB = dict(n_scenarios=int(len(anchors)),
                  success_rate=float(succB.mean()),
                  mean_err_cells=float(errsB.mean()),
                  median_err_cells=float(np.median(errsB)),
                  fail_rate_low_distinct=float((~succB[lo]).mean()) if lo.any() else -1.0,
                  fail_rate_high_distinct=float((~succB[hi]).mean()) if hi.any() else -1.0)
    print(f'[bench B] {len(anchors)} distinct scenarios: SUCCESS {succB.mean()*100:.0f}%  '
          f'median err {np.median(errsB):.2f} cells  |  '
          f'fail low-distinct {blockB["fail_rate_low_distinct"]*100:.0f}% vs '
          f'high {blockB["fail_rate_high_distinct"]*100:.0f}%')

    # ---- plots ----------------------------------------------------------#
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(errsA, bins=20, color='tab:blue', alpha=0.85)
    ax[0].axvline(args.success_tol_cells, color='r', ls='--', label='success tol')
    ax[0].set_title(f'A: standard scenario x{args.n_runs}\nsuccess {succA.mean()*100:.0f}/100')
    ax[0].set_xlabel('final localization error (cells)'); ax[0].set_ylabel('count'); ax[0].legend()
    ax[1].hist(errsB, bins=20, color='tab:green', alpha=0.85)
    ax[1].axvline(args.success_tol_cells, color='r', ls='--')
    ax[1].set_title(f'B: {len(anchors)} distinct scenarios\nsuccess {succB.mean()*100:.0f}%')
    ax[1].set_xlabel('final localization error (cells)'); ax[1].set_ylabel('count')
    fig.suptitle(f'LiDAR grid localization benchmark (K={args.K}, tol {args.success_tol_cells} cells)')
    fig.tight_layout(); fig.savefig(f'{args.out}/benchmark_hist.png', dpi=120); plt.close(fig)

    summary = dict(env=args.env, K=args.K, hist_stride=args.hist_stride,
                   success_tol_cells=args.success_tol_cells, xy_step=args.xy_step,
                   n_theta=args.n_theta, sigma=args.sigma,
                   block_A_consistency=blockA, block_B_coverage=blockB)
    with open(f'{args.out}/benchmark.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'[bench] saved benchmark.json + benchmark_hist.png -> {args.out}')


if __name__ == '__main__':
    main()
