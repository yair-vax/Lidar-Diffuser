"""Offline roundtrip test for the grid navigator (CPU, numpy only -- no torch/mujoco).

Closes the loop grid -> route -> scans -> grid BEFORE any MuJoCo rollout:

  1. build/load the (x, y, theta) reference grid;
  2. synthesize routes between random far free-cell pairs (RouteScanSynthesizer --
     the exact code the rollout will execute);
  3. checks: every route point in free space; scans valid; the DECODED route
     (localize_sequence on the synthesized scans) must end at the goal cell and
     track the BFS route -- i.e. the scan plan we hand the executor is one the
     localizer itself recognizes as "start -> goal";
  4. probe + band_along_route sanity.

Run (repo root, any machine):
    python diffuser/pl_eval/lidar_eval/lidar_route_nav_test.py \
        --env antmaze-giant-stitch-v0 --n_routes 20
Exit code 0 == PASS.
"""

import argparse
import importlib.util
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIDAR_DIR = os.path.abspath(os.path.join(HERE, '..', '..', 'datasets', 'lidar'))


def _load(name):
    """File-path import (no ``diffuser`` package -> no torch on bare CPU boxes)."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(LIDAR_DIR, f'{name}.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


lidar_grid = _load('lidar_grid')
lidar_localization = _load('lidar_localization')
lidar_route_nav = _load('lidar_route_nav')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', default='antmaze-giant-stitch-v0')
    ap.add_argument('--n_routes', type=int, default=20)
    ap.add_argument('--xy_step', type=float, default=1.0)
    ap.add_argument('--n_theta', type=int, default=12)
    ap.add_argument('--n_decode', type=int, default=40, help='scans decoded per route')
    ap.add_argument('--min_dist_cells', type=float, default=15.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--cache_root', default='data/ogb_maze/lidar_grid')
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    t0 = time.time()
    grid = lidar_grid.load_or_build_grid(
        args.env, dict(n_beams=30, max_range=12.0),
        xy_step=args.xy_step, n_theta=args.n_theta, cache_root=args.cache_root)
    loc = lidar_localization.GridLocalizer(grid, sigma=1.0)
    sc = grid.scanner
    synth = lidar_route_nav.RouteScanSynthesizer(sc, step_wu=0.13, yaw_smooth_win=31)
    print(f'[setup] grid {grid.n_poses} poses in {time.time() - t0:.1f}s')

    free_ij = np.argwhere(sc.maze_map == 0)
    fails, end_errs, track_errs, lens = [], [], [], []

    for k in range(args.n_routes):
        ## random far pair of free cells
        for _ in range(200):
            a, b = free_ij[rng.integers(len(free_ij))], free_ij[rng.integers(len(free_ij))]
            if np.abs(a - b).sum() >= args.min_dist_cells:
                break
        sxy = np.asarray(synth.cell_center(*a)) + rng.uniform(-1.0, 1.0, 2)
        gxy = np.asarray(synth.cell_center(*b))
        route = synth.synthesize(sxy, gxy, start_yaw=float(rng.uniform(-np.pi, np.pi)),
                                 goal_hold=10)
        if route is None:
            fails.append((k, 'no route')); continue

        xy, scans = route['xy'], route['scans']
        lens.append(len(xy))
        ## (a) free-space + scan validity
        i, j = sc.world_to_cell(xy[:, 0], xy[:, 1])
        if np.any(sc._is_wall(i, j)):
            fails.append((k, 'route hits wall')); continue
        if not np.all(np.isfinite(scans)) or scans.min() < 0 or scans.max() > 12.0 + 1e-5:
            fails.append((k, 'bad scan values')); continue

        ## (b) decode the synthesized plan with the motion-history filter
        sel = np.linspace(0, len(scans) - 1, min(args.n_decode, len(scans))).astype(int)
        r = loc.localize_sequence(scans[sel], motion_sigma_cells=2.0, theta_blur=1)
        dec_xy = r['map_xy']
        ## end-cell error (in cells) vs the intended goal
        ge = np.abs(np.asarray(grid.cell_of_xy(*dec_xy[-1])) -
                    np.asarray(grid.cell_of_xy(*gxy))).sum()
        end_errs.append(float(ge))
        ## median tracking error along the route (cells)
        te = np.linalg.norm(dec_xy - xy[sel], axis=1) / sc.maze_unit
        track_errs.append(float(np.median(te)))

    ## (c) probe sanity
    a = free_ij[rng.integers(len(free_ij))]
    pxy = np.asarray(synth.cell_center(*a))
    pscan = sc.scan(pxy[0], pxy[1], 0.3)
    probe = synth.probe(pxy, 0.3, pscan, goal_hold=5)
    ok_probe = (probe['scans'].ndim == 2 and probe['scans'].shape[0] >= 2
                and np.all(np.isfinite(probe['scans'])))

    ## (d) band_along_route sanity (K=8, step 2.0 wu -- the gband setting)
    route = synth.synthesize(pxy, np.asarray(synth.cell_center(*free_ij[0])), goal_hold=0)
    ok_band = True
    if route is not None:
        band = synth.band_along_route(route, k=8, band_step_wu=2.0)
        ok_band = band.shape == (8, sc.n_beams) and np.all(np.isfinite(band))

    n_ok = args.n_routes - len(fails)
    med_end = float(np.median(end_errs)) if end_errs else np.inf
    med_trk = float(np.median(track_errs)) if track_errs else np.inf
    frac_end_ok = float(np.mean(np.asarray(end_errs) <= 1.0)) if end_errs else 0.0

    print(f'\n[routes]  synthesized ok: {n_ok}/{args.n_routes} '
          f'(median plan len {int(np.median(lens)) if lens else 0} frames)')
    print(f'[decode]  end-cell err: median {med_end:.1f} cells, <=1 cell in '
          f'{100 * frac_end_ok:.0f}% | route tracking median {med_trk:.2f} cells')
    print(f'[probe]   ok={ok_probe}   [band] ok={ok_band}')
    for f in fails:
        print(f'  fail: route {f[0]}: {f[1]}')

    passed = (n_ok >= max(1, int(0.9 * args.n_routes)) and med_end <= 1.0
              and frac_end_ok >= 0.7 and med_trk <= 1.0 and ok_probe and ok_band)
    print(f'\n{"PASS" if passed else "FAIL"}: grid->route->scans->grid roundtrip '
          f'({time.time() - t0:.1f}s total)')
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
