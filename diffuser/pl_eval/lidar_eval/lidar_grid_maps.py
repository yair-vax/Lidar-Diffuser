#!/usr/bin/env python
"""
Stage-2 maps for the (x, y, theta) LiDAR localization grid  (PM plan, Algorithm & Analysis).

Builds the reference grid, then for a chosen **target** pose produces the three maps
the spec asks for, all overlaid on the maze:

  1. ``distance_to_target_map.png`` -- the headline "distance-difference map": per (x, y),
     the LiDAR-scan distance from that location's best-orientation scan to the target's
     scan. Bright, far-away minima are the *look-alikes* that make a single goal scan
     ambiguous (exactly the failure the diagnosis traced).
  2. ``probability_map.png`` -- the measurement-model posterior over position for the
     target scan (probability mapping: uncertainty -> likelihood). Multiple modes ==
     the target scan cannot pin one location.
  3. ``distinctiveness_map.png`` -- whole-grid localizability: per (x, y), scan distance
     to the nearest *far* pose. Dark == locally ambiguous (open rooms / symmetric
     corridors), the structure-vs-difficulty signal.

Also writes ``target_report.json`` (target pose, #look-alikes at tolerances, nearest
look-alike + its distance in cells).

Pure numpy + matplotlib; no torch / MuJoCo. Runs in seconds.

Example::

    python -m diffuser.pl_eval.lidar_eval.lidar_grid_maps \
        --env antmaze-giant-stitch-v0 --xy_step 1.0 --n_theta 12 \
        --out logs/lidar_eval/grid_giant
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lidar_grid_common as C  # noqa: E402  (torch-free bootstrap, same dir)
LG, LL = C.LG, C.LL
add_grid_args = C.add_grid_args
lidar_config_from_args = C.lidar_config_from_args
plot_xy_map = C.plot_xy_map


def _default_target_cell(scanner):
    """Top-right-most free maze cell (a representative giant-maze goal region)."""
    free = np.argwhere(scanner.maze_map == 0)          # (i, j)
    k = np.argmax(free[:, 0] + free[:, 1])
    return int(free[k, 0]), int(free[k, 1])


def _cell_to_xy(scanner, i, j):
    return j * scanner.maze_unit - scanner.offset_x, i * scanner.maze_unit - scanner.offset_y


def main():
    ap = add_grid_args(argparse.ArgumentParser())
    ap.add_argument('--target_xy', type=float, nargs=2, default=None,
                    help='target world (x y); overrides --target_cell')
    ap.add_argument('--target_cell', type=int, nargs=2, default=None,
                    help='target maze cell (i j); default = top-right free cell')
    ap.add_argument('--target_yaw', type=float, default=0.0,
                    help='target heading (rad); eval default 0.0')
    ap.add_argument('--tols', type=float, nargs='+', default=[0.25, 0.5, 1.0],
                    help='per-beam RMS look-alike tolerances (world units)')
    ap.add_argument('--min_cell_gap', type=float, default=3.0)
    ap.add_argument('--out', default='logs/lidar_eval/grid_maps')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    grid = LG.load_or_build_grid(args.env, lidar_config_from_args(args),
                                 xy_step=args.xy_step, n_theta=args.n_theta,
                                 cache_root=args.cache_root)
    loc = LL.GridLocalizer(grid, sigma=args.sigma)
    scanner = grid.scanner
    mu = grid.maze_unit

    # -- resolve the target pose ------------------------------------------#
    if args.target_xy is not None:
        tx, ty = float(args.target_xy[0]), float(args.target_xy[1])
        ti, tj = grid.cell_of_xy(tx, ty)
    else:
        ti, tj = args.target_cell if args.target_cell is not None \
            else _default_target_cell(scanner)
        tx, ty = _cell_to_xy(scanner, ti, tj)
    target_scan = scanner.scan(tx, ty, args.target_yaw).astype(np.float32)
    print(f'[maps] target cell=({ti},{tj}) world=({tx:.1f},{ty:.1f}) '
          f'yaw={args.target_yaw:.2f}')

    tmark = dict(xy=(tx, ty), c='red', m='*', s=260, label='target')

    # -- target look-alike stats (computed first so maps can mark the decoy) #
    # per free (x, y): best-orientation scan distance to the target scan.
    d_all = LL.scan_distance(target_scan, grid.free_scans)[0]     # (M,)
    # reduce to per-(x,y) best over theta
    per_xy = {}
    for k in range(grid.free_scans.shape[0]):
        key = (int(grid.free_rc[k, 0]), int(grid.free_rc[k, 1]))
        per_xy[key] = min(per_xy.get(key, np.inf), d_all[k])
    keys = list(per_xy.keys())
    kxy = np.array([[grid.x_vals[c], grid.y_vals[r]] for (r, c) in keys])
    kd = np.array([per_xy[k] for k in keys])
    cell_gap = np.sqrt(((kxy - np.array([tx, ty])) ** 2).sum(1)) / mu
    far = cell_gap >= args.min_cell_gap
    tol_stats = {}
    for tol in args.tols:
        look = far & (kd < tol)
        tol_stats[f'{tol}'] = dict(
            n_lookalikes=int(look.sum()),
            nearest_lookalike_cells=float(cell_gap[look].min()) if look.any() else -1.0)

    # nearest FAR look-alike (smallest scan distance among far cells) -- the
    # decoy the goal scan is most easily confused with (excludes the self-match).
    if far.any():
        kfar = np.where(far)[0]
        b = kfar[int(np.argmin(kd[kfar]))]
        far_xy = [float(kxy[b, 0]), float(kxy[b, 1])]
        far_dist = float(kd[b]); far_cells = float(cell_gap[b])
    else:
        far_xy, far_dist, far_cells = [float(tx), float(ty)], 0.0, 0.0
    dmark = dict(xy=(far_xy[0], far_xy[1]), c='deepskyblue', m='X', s=180,
                 label='nearest decoy')

    # -- 1. distance-difference map (headline) ----------------------------#
    dist_xy, _ = loc.distance_map_to_target(target_scan, reduce='min')
    plot_xy_map(grid, dist_xy, f'{args.out}/distance_to_target_map.png',
                'Distance to target scan  (min over orientation)',
                cmap='viridis', label='per-beam RMS distance (world units)',
                marks=[tmark, dmark])

    # -- 2. probability (posterior) map -----------------------------------#
    post = loc.posterior_xy(target_scan)
    plot_xy_map(grid, post, f'{args.out}/probability_map.png',
                'Measurement posterior for target scan  (theta marginalized)',
                cmap='magma', label='posterior probability', marks=[tmark, dmark])

    # -- 3. distinctiveness / ambiguity map -------------------------------#
    distinct = loc.distinctiveness_map(min_cell_gap=args.min_cell_gap, reduce='mean')
    plot_xy_map(grid, distinct, f'{args.out}/distinctiveness_map.png',
                'Localizability  (scan dist to nearest far pose; dark = ambiguous)',
                cmap='cividis', label='nearest-far-pose distance (world units)')

    report = dict(env=args.env, target_cell=[ti, tj], target_xy=[tx, ty],
                  target_yaw=args.target_yaw, xy_step=args.xy_step,
                  n_theta=args.n_theta, sigma=args.sigma, n_beams=args.n_beams,
                  max_range=args.max_range,
                  nearest_far_lookalike_xy=far_xy,
                  nearest_far_lookalike_distance=far_dist,
                  nearest_far_lookalike_cells=far_cells,
                  lookalikes_by_tol=tol_stats,
                  grid_free_xy=int(grid.n_free_xy), grid_poses=int(grid.n_poses))
    with open(f'{args.out}/target_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print(f'[maps] look-alikes: ' +
          '  '.join(f'tol{t}: {tol_stats[str(t)]["n_lookalikes"]}' for t in args.tols))
    print(f'[maps] saved 3 maps + target_report.json -> {args.out}')


if __name__ == '__main__':
    main()
