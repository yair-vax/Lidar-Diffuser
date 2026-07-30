"""
Standalone tool to materialize a LiDAR-format copy of an OGBench maze dataset.

This implements stage 3 of the project spec ("dataset management: keep an exact
copy of the dataset in the dedicated LiDAR-scan format"). Training does NOT
require running this first -- the dataset loader computes & caches LiDAR scans on
the fly -- but this script is useful to (a) pre-generate / inspect the LiDAR data,
and (b) produce a portable, self-contained ``.npz`` that other tools (e.g. the
evaluation histogram script) can consume without OGBench installed.

Two input modes:

  1. From a local ``.npz`` (e.g. the bundled luotest files), no OGBench needed:
        python -m diffuser.datasets.lidar.gen_lidar_dataset \
            --npz data/ogb_maze/antmaze-giant-stitch-v0-luotest.npz \
            --env antmaze-giant-stitch-v0 \
            --out data/ogb_maze/lidar/antmaze-giant-stitch-v0-luotest_lidar.npz

  2. From the full OGBench dataset (requires the `ogbench` conda env):
        python -m diffuser.datasets.lidar.gen_lidar_dataset \
            --env antmaze-giant-stitch-v0 \
            --out data/ogb_maze/lidar/antmaze-giant-stitch-v0_lidar.npz

Output npz keys:
    lidar       (N, n_beams)   float32  -- the LiDAR observation (planner input)
    xyo_lidar   (N, n_beams+4) float32  -- [x, y, cos(yaw), sin(yaw), lidar] (invdyn)
    xy          (N, 2)         float32  -- ground-truth position (eval only)
    yaw         (N,)           float32  -- ground-truth heading  (eval only)
    actions     (N, A)         float32
    terminals   (N,)           bool
    <plus scalar metadata: n_beams, max_range, fov, step, maze_unit, env_name>
"""

import os
## Force EGL rendering for the --env mode (loads OGBench/mujoco on a headless
## NVIDIA server). Harmless for the local-npz mode. Set before any mujoco import.
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('MUJOCO_GL', 'egl')

import argparse
import numpy as np

from diffuser.datasets.lidar.lidar_sim import (
    make_scanner_for_env, obs_to_xy_yaw,
)


def _load_raw(npz_path, env_name):
    """Return (observations, actions, terminals) either from a local npz or by
    loading the full OGBench dataset for ``env_name``."""
    if npz_path is not None:
        d = np.load(npz_path)
        obs = d['observations']
        act = d['actions']
        term = d['terminals'] if 'terminals' in d else np.zeros(len(obs), dtype=bool)
        return obs, act, term
    # Lazy OGBench import so the local-npz mode does not need OGBench / mujoco
    # installed (the hardest part of the environment to set up).
    import ogbench  # noqa: F401
    from ogbench.utils import ogb_load_dataset
    wrapped = ogbench.make_env_and_datasets(env_name, env_only=True)
    env = wrapped.unwrapped
    env.name = env_name
    env.dset_h5path = None
    ds = ogb_load_dataset(env)
    return ds['observations'], ds['actions'], ds['terminals']


def main():
    p = argparse.ArgumentParser(description='Generate a LiDAR-format dataset copy.')
    p.add_argument('--env', required=True, help='OGBench env name, e.g. antmaze-giant-stitch-v0')
    p.add_argument('--npz', default=None, help='optional local .npz to read instead of OGBench')
    p.add_argument('--out', required=True, help='output .npz path')
    p.add_argument('--n_beams', type=int, default=30)
    p.add_argument('--max_range', type=float, default=12.0)
    p.add_argument('--fov', type=float, default=2.0 * np.pi)
    p.add_argument('--step', type=float, default=0.1)
    args = p.parse_args()

    obs, act, term = _load_raw(args.npz, args.env)
    print(f'[gen_lidar] loaded {obs.shape[0]} transitions, obs_dim={obs.shape[1]}')

    lidar_config = dict(n_beams=args.n_beams, max_range=args.max_range,
                        fov=args.fov, step=args.step)
    scanner = make_scanner_for_env(args.env, lidar_config)

    has_quat = obs.shape[1] >= 7
    print(f'[gen_lidar] casting LiDAR ({args.n_beams} beams, range {args.max_range}) ...')
    lidar = scanner.scan_obs(obs, has_quat=has_quat)

    xy, yaw = obs_to_xy_yaw(obs, has_quat=has_quat)
    xyo_lidar = np.concatenate(
        [xy, np.cos(yaw)[:, None], np.sin(yaw)[:, None], lidar], axis=1).astype(np.float32)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        args.out,
        lidar=lidar.astype(np.float32),
        xyo_lidar=xyo_lidar,
        xy=xy.astype(np.float32),
        yaw=yaw.astype(np.float32),
        actions=act.astype(np.float32),
        terminals=term.astype(bool),
        env_name=np.array(args.env),
        n_beams=np.array(args.n_beams),
        max_range=np.array(args.max_range, dtype=np.float32),
        fov=np.array(args.fov, dtype=np.float32),
        step=np.array(args.step, dtype=np.float32),
        maze_unit=np.array(scanner.maze_unit, dtype=np.float32),
        offset_x=np.array(scanner.offset_x, dtype=np.float32),
        offset_y=np.array(scanner.offset_y, dtype=np.float32),
    )
    print(f'[gen_lidar] wrote {args.out}')
    print(f'[gen_lidar]   lidar={lidar.shape}, xyo_lidar={xyo_lidar.shape}')
    print(f'[gen_lidar]   dist min/mean/max = '
          f'{lidar.min():.2f}/{lidar.mean():.2f}/{lidar.max():.2f}')


if __name__ == '__main__':
    main()
