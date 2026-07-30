#!/usr/bin/env python
"""Goal-scan ambiguity + max_range pre-check  (no models, no MuJoCo, no dataset).

Decides whether **Option 2** (a longer-``max_range`` goal scan) can even work,
BEFORE paying for the lidar re-cache + planner & inv-dyn retrain.

The eval-time goal is a single 30-beam LiDAR scan rendered at the goal cell
(yaw = ``lidar_goal_yaw``, default 0.0). If many OTHER maze cells produce a
nearly identical scan, that one scan cannot pin the goal -> the plan routes to a
look-alike corridor (the observed ~8-12 cell miss). This script quantifies that
"look-alike" set over ALL free cells, at several ``max_range`` values, using the
exact same scanner that builds the training/eval scans. Pure numpy, ~seconds.

READ THE RESULT:
  * look-alike count / %-ambiguous DROPS sharply as max_range grows
        -> longer range disambiguates -> Option 2 is viable, fund the retrain.
  * stays high / flat
        -> longer range will NOT disambiguate -> do NOT fund Option 2;
           use a richer goal rep (wide-baseline goal sequence) instead.

Run:
    python diffuser/pl_eval/lidar_eval/goal_scan_ambiguity.py \
        --env antmaze-giant-stitch-v0 --ranges 12 24 30 \
        --out logs/lidar_eval/goal_scan_ambiguity.json
"""
import argparse, json, os, importlib.util
import numpy as np


def _load_make_scanner():
    """Prefer the package import; fall back to loading the pure-numpy scanner
    module directly so this check runs on a bare laptop (the ``diffuser``
    package __init__ pulls in torch, which this script does not need)."""
    try:
        from diffuser.datasets.lidar.lidar_sim import make_scanner_for_env
        return make_scanner_for_env
    except Exception:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.normpath(os.path.join(here, '..', '..', 'datasets', 'lidar', 'lidar_sim.py'))
        spec = importlib.util.spec_from_file_location('lidar_sim', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.make_scanner_for_env


make_scanner_for_env = _load_make_scanner()


def free_cell_world_xy(scanner):
    """(M,2) world centres of every free cell + (M,2) int cell indices (i, j).
    Inverts LidarScanner.world_to_cell: x = j*unit - off_x, y = i*unit - off_y."""
    free = np.argwhere(scanner.maze_map == 0)            # (M, 2) -> (i, j)
    i, j = free[:, 0], free[:, 1]
    x = j * scanner.maze_unit - scanner.offset_x
    y = i * scanner.maze_unit - scanner.offset_y
    return np.stack([x, y], 1).astype(np.float64), free


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', default='antmaze-giant-stitch-v0')
    ap.add_argument('--ranges', type=float, nargs='+', default=[12.0, 24.0, 30.0])
    ap.add_argument('--n_beams', type=int, default=30)
    ap.add_argument('--yaw', type=float, default=0.0,
                    help='goal-scan heading (eval default 0.0)')
    ap.add_argument('--min_cell_gap', type=float, default=3.0,
                    help='ignore look-alikes closer than this many cells (true neighbours)')
    ap.add_argument('--tols', type=float, nargs='+', default=[0.25, 0.5, 1.0],
                    help='per-beam RMS match tolerances, world units (1 cell = maze_unit)')
    ap.add_argument('--out', default='logs/lidar_eval/goal_scan_ambiguity.json')
    args = ap.parse_args()

    report = {'env': args.env, 'yaw': args.yaw, 'min_cell_gap': args.min_cell_gap,
              'tols': args.tols, 'ranges': {}}

    for R in args.ranges:
        scanner = make_scanner_for_env(
            args.env, dict(n_beams=args.n_beams, max_range=float(R), fov=2.0 * np.pi))
        xy, cells = free_cell_world_xy(scanner)
        M = xy.shape[0]
        yaw = np.full(M, float(args.yaw))
        scans = scanner.scan_batch(xy, yaw)              # (M, n_beams) world units

        # cell-grid distance (in cells) between every pair of free cells
        cd = np.sqrt(((cells[:, None, :] - cells[None, :, :]) ** 2).sum(-1))  # (M, M)
        far = cd >= args.min_cell_gap                    # exclude self + true neighbours

        # per-beam RMS scan distance, chunked over goal rows to bound memory
        rms = np.empty((M, M), dtype=np.float32)
        CH = 64
        for a in range(0, M, CH):
            d = scans[a:a + CH, None, :] - scans[None, :, :]   # (<=CH, M, n_beams)
            rms[a:a + CH] = np.sqrt((d ** 2).mean(-1))

        per_tol = {}
        for tol in args.tols:
            look = far & (rms < tol)                     # (M, M) look-alike mask
            n_look = look.sum(1)                          # per-goal look-alike count
            nd = np.where(look, cd, np.inf).min(1)        # nearest look-alike (cells)
            finite = np.isfinite(nd)
            per_tol[f'{tol}'] = dict(
                mean_lookalikes=float(n_look.mean()),
                median_lookalikes=float(np.median(n_look)),
                max_lookalikes=int(n_look.max()),
                frac_cells_ambiguous=float((n_look > 0).mean()),
                median_nearest_lookalike_cells=float(np.median(nd[finite]) if finite.any() else -1.0),
            )
        report['ranges'][f'{R}'] = dict(n_free_cells=int(M), per_tol=per_tol)

        print(f'\n=== max_range={R}  ({M} free cells, yaw={args.yaw}) ===')
        for tol, dct in per_tol.items():
            print(f'  rms<{tol:>4}:  ambiguous {dct["frac_cells_ambiguous"]*100:5.1f}%  '
                  f'mean look-alikes {dct["mean_lookalikes"]:6.1f}  '
                  f'median nearest {dct["median_nearest_lookalike_cells"]:.1f} cells')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'\nsaved -> {args.out}')
    print('READ: %ambiguous / mean look-alikes DROP sharply across ranges -> Option 2 viable; '
          'flat -> longer range will NOT disambiguate the goal.')


if __name__ == '__main__':
    main()
