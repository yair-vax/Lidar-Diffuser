"""
3D (x, y, theta) LiDAR reference grid for the OGBench maze environments.

This implements **Stage 1 of the PM plan (Grid Construction & Mapping)**: discretize
the maze into a grid over the ``X``, ``Y`` and ``theta`` (orientation) axes, simulate
a LiDAR scan at *every* free grid pose, and store the result as a
``Rows x Columns x Orientation x n_beams`` tensor (one "map copy" per orientation
angle).  Angular resolution defaults to 30 degrees (12 orientation bins), matching the
spec's "~30 degrees per point".

It is a thin, cached layer on top of the pure-numpy :class:`LidarScanner`
(``diffuser/datasets/lidar/lidar_sim.py``), so it has **no torch / mujoco / ogbench
dependency** and runs on any machine in seconds (the whole giant grid at a 1-world-unit
spatial step + 12 orientations is a few thousand scans -> well under the "few minutes
per run" budget in the spec).

The two views it exposes are:

* **Dense grid** (for maps / visualization) --
      ``scans_grid``   : ``(n_row, n_col, n_theta, n_beams)`` float32, wall cells = NaN
      ``free_mask_xy`` : ``(n_row, n_col)`` bool, True where the (x, y) is free space
      ``x_vals``       : ``(n_col,)`` world x of each column
      ``y_vals``       : ``(n_row,)`` world y of each row
      ``theta_vals``   : ``(n_theta,)`` orientation of each map copy (radians)

* **Flat free-pose list** (for fast localization / nearest-scan search) --
      ``free_scans``   : ``(M, n_beams)`` float32, one row per free (x, y, theta) pose
      ``free_xy``      : ``(M, 2)``     world position of each pose
      ``free_theta``   : ``(M,)``       orientation of each pose
      ``free_rc``      : ``(M, 2)``     (row, col) index into the dense grid
      ``free_ti``      : ``(M,)``       theta index into the dense grid

Both are produced from the same ray-casting pass, so a pose in the flat list and its
``(row, col, theta)`` slot in the dense grid carry identical scans.
"""

import os
import importlib.util
import numpy as np


def _import_lidar_sim():
    """Import the pure-numpy LiDAR simulator.

    Prefer the normal package import (used by the torch rollout). Fall back to
    loading the sibling ``lidar_sim.py`` by file path so this module also works
    on a bare CPU box where importing the ``diffuser`` package would pull in
    torch/mujoco (mirrors the fallback in the lidar_eval CLI scripts)."""
    try:
        from diffuser.datasets.lidar import lidar_sim as _ls
        return _ls
    except Exception:  # noqa: BLE001
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, 'lidar_sim.py')
        spec = importlib.util.spec_from_file_location('lidar_sim', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


_ls = _import_lidar_sim()
make_scanner_for_env = _ls.make_scanner_for_env
LidarScanner = _ls.LidarScanner


class LidarGrid:
    """A cached grid of simulated LiDAR scans over the (x, y, theta) free space.

    Args:
        scanner: a built :class:`LidarScanner` (carries the maze geometry + beams).
        xy_step: spatial resolution in world units (1 maze cell == ``maze_unit``,
            default 4.0). Smaller = finer localization, more poses. Default 1.0
            (a quarter cell) which is plenty and still tiny.
        n_theta: number of orientation bins over [0, 2*pi). Default 12 == 30 deg/bin
            (the spec's "~30 degrees per point").
        theta0: orientation of the first bin (radians).
        margin: extra world-unit padding around the free bounding box when laying out
            the regular (x, y) lattice (keeps the border cells sampled).
    """

    def __init__(self, scanner, xy_step=1.0, n_theta=12, theta0=0.0, margin=0.0):
        assert isinstance(scanner, LidarScanner)
        self.scanner = scanner
        self.xy_step = float(xy_step)
        self.n_theta = int(n_theta)
        self.theta0 = float(theta0)
        self.margin = float(margin)

        self.theta_vals = (self.theta0 +
                           np.arange(self.n_theta) * (2.0 * np.pi / self.n_theta)).astype(np.float64)

        # regular world lattice over the free bounding box -------------------#
        free_ij = np.argwhere(scanner.maze_map == 0)          # (Ncell, 2) -> (i, j)
        fx = free_ij[:, 1] * scanner.maze_unit - scanner.offset_x
        fy = free_ij[:, 0] * scanner.maze_unit - scanner.offset_y
        x_lo, x_hi = fx.min() - self.margin, fx.max() + self.margin
        y_lo, y_hi = fy.min() - self.margin, fy.max() + self.margin
        self.x_vals = np.arange(x_lo, x_hi + 1e-6, self.xy_step, dtype=np.float64)
        self.y_vals = np.arange(y_lo, y_hi + 1e-6, self.xy_step, dtype=np.float64)
        self.n_col = len(self.x_vals)
        self.n_row = len(self.y_vals)

        # which lattice points are in a free cell ---------------------------#
        XX, YY = np.meshgrid(self.x_vals, self.y_vals)        # (n_row, n_col)
        ci, cj = scanner.world_to_cell(XX.ravel(), YY.ravel())
        free = ~scanner._is_wall(ci, cj)                      # (n_row*n_col,)
        self.free_mask_xy = free.reshape(self.n_row, self.n_col)

        # populated by build()
        self.scans_grid = None       # (n_row, n_col, n_theta, n_beams)
        self.free_scans = None       # (M, n_beams)
        self.free_xy = None          # (M, 2)
        self.free_theta = None       # (M,)
        self.free_rc = None          # (M, 2)
        self.free_ti = None          # (M,)

    # -- properties --------------------------------------------------------#
    @property
    def n_beams(self):
        return self.scanner.n_beams

    @property
    def maze_unit(self):
        return self.scanner.maze_unit

    @property
    def n_free_xy(self):
        return int(self.free_mask_xy.sum())

    @property
    def n_poses(self):
        return self.n_free_xy * self.n_theta

    # -- build -------------------------------------------------------------#
    def build(self, verbose=True):
        """Ray-cast a LiDAR scan at every free (x, y, theta) pose.

        Fills both the dense ``scans_grid`` (wall cells left as NaN) and the flat
        free-pose arrays. Vectorized over all free (x, y) at each theta.
        """
        rr, cc = np.where(self.free_mask_xy)                  # free row/col indices
        M_xy = len(rr)
        xy = np.stack([self.x_vals[cc], self.y_vals[rr]], axis=1)   # (M_xy, 2)

        nb = self.n_beams
        self.scans_grid = np.full((self.n_row, self.n_col, self.n_theta, nb),
                                  np.nan, dtype=np.float32)

        free_scans, free_xy, free_theta, free_rc, free_ti = [], [], [], [], []
        for ti, th in enumerate(self.theta_vals):
            yaw = np.full(M_xy, th, dtype=np.float64)
            scans = self.scanner.scan_batch(xy, yaw)          # (M_xy, nb)
            self.scans_grid[rr, cc, ti, :] = scans
            free_scans.append(scans)
            free_xy.append(xy)
            free_theta.append(yaw.astype(np.float32))
            free_rc.append(np.stack([rr, cc], axis=1))
            free_ti.append(np.full(M_xy, ti, dtype=np.int64))
            if verbose:
                print(f'  [grid] theta {ti + 1}/{self.n_theta} '
                      f'({np.degrees(th):.0f} deg): {M_xy} scans')

        self.free_scans = np.concatenate(free_scans, 0).astype(np.float32)
        self.free_xy = np.concatenate(free_xy, 0).astype(np.float32)
        self.free_theta = np.concatenate(free_theta, 0).astype(np.float32)
        self.free_rc = np.concatenate(free_rc, 0).astype(np.int64)
        self.free_ti = np.concatenate(free_ti, 0).astype(np.int64)
        if verbose:
            print(f'  [grid] built {self.free_scans.shape[0]} poses '
                  f'({M_xy} free xy x {self.n_theta} theta), n_beams={nb}')
        return self

    # -- geometry helpers --------------------------------------------------#
    def xy_to_rc(self, x, y):
        """Nearest (row, col) lattice index for a world (x, y)."""
        c = int(np.clip(round((x - self.x_vals[0]) / self.xy_step), 0, self.n_col - 1))
        r = int(np.clip(round((y - self.y_vals[0]) / self.xy_step), 0, self.n_row - 1))
        return r, c

    def cell_of_xy(self, x, y):
        """OGBench integer maze cell (i, j) of a world (x, y)."""
        i, j = self.scanner.world_to_cell(np.asarray([x]), np.asarray([y]))
        return int(i[0]), int(j[0])

    # -- persistence -------------------------------------------------------#
    def save(self, path):
        assert self.free_scans is not None, 'call build() before save()'
        os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
        np.savez_compressed(
            path,
            # config
            xy_step=self.xy_step, n_theta=self.n_theta, theta0=self.theta0,
            margin=self.margin,
            scanner_cfg=np.array(_encode_cfg(self.scanner.config_dict())),
            maze_map=self.scanner.maze_map,
            # dense grid
            x_vals=self.x_vals, y_vals=self.y_vals, theta_vals=self.theta_vals,
            free_mask_xy=self.free_mask_xy, scans_grid=self.scans_grid,
            # flat poses
            free_scans=self.free_scans, free_xy=self.free_xy,
            free_theta=self.free_theta, free_rc=self.free_rc, free_ti=self.free_ti,
        )
        return path

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=True)
        cfg = _decode_cfg(d['scanner_cfg'].item() if d['scanner_cfg'].ndim == 0
                          else str(d['scanner_cfg']))
        scanner = LidarScanner(maze_map=d['maze_map'], **cfg)
        obj = cls(scanner, xy_step=float(d['xy_step']), n_theta=int(d['n_theta']),
                  theta0=float(d['theta0']), margin=float(d['margin']))
        obj.x_vals = d['x_vals']; obj.y_vals = d['y_vals']
        obj.theta_vals = d['theta_vals']; obj.free_mask_xy = d['free_mask_xy']
        obj.n_row, obj.n_col = obj.free_mask_xy.shape
        obj.scans_grid = d['scans_grid']
        obj.free_scans = d['free_scans']; obj.free_xy = d['free_xy']
        obj.free_theta = d['free_theta']; obj.free_rc = d['free_rc']
        obj.free_ti = d['free_ti']
        return obj


def _encode_cfg(cfg):
    """LidarScanner.config_dict -> the kwargs LidarScanner.__init__ accepts."""
    keep = ('n_beams', 'max_range', 'fov', 'fov_centered', 'maze_unit',
            'offset_x', 'offset_y', 'step')
    return repr({k: cfg[k] for k in keep if k in cfg})


def _decode_cfg(s):
    import ast
    return ast.literal_eval(s if isinstance(s, str) else s.decode())


def build_grid_for_env(env_name, lidar_config=None, xy_step=1.0, n_theta=12,
                       theta0=0.0, margin=0.0, verbose=True):
    """Convenience: build (and return, unsaved) a :class:`LidarGrid` for an env name.

    ``lidar_config`` is the same dict the dataset configs use (n_beams, max_range,
    fov, ...); defaults match the trained model (30 beams, max_range 12, 360 deg).
    """
    lidar_config = dict(lidar_config or {})
    lidar_config.setdefault('n_beams', 30)
    lidar_config.setdefault('max_range', 12.0)
    lidar_config.setdefault('fov', 2.0 * np.pi)
    scanner = make_scanner_for_env(env_name, lidar_config)
    grid = LidarGrid(scanner, xy_step=xy_step, n_theta=n_theta,
                     theta0=theta0, margin=margin)
    if verbose:
        print(f'[grid] env={env_name} lattice={grid.n_row}x{grid.n_col} '
              f'({grid.n_free_xy} free xy) x {n_theta} theta '
              f'= {grid.n_poses} poses; xy_step={xy_step} '
              f'(1 cell={grid.maze_unit} wu)')
    return grid.build(verbose=verbose)


def default_cache_path(env_name, xy_step, n_theta, max_range, root='data/ogb_maze/lidar_grid'):
    tag = f'{env_name}_xy{xy_step:g}_th{n_theta}_mr{max_range:g}.npz'
    return os.path.join(root, tag)


def load_or_build_grid(env_name, lidar_config=None, xy_step=1.0, n_theta=12,
                       cache_root='data/ogb_maze/lidar_grid', verbose=True):
    """Load the cached grid for these settings, or build + cache it on first use."""
    lidar_config = dict(lidar_config or {})
    max_range = lidar_config.get('max_range', 12.0)
    path = default_cache_path(env_name, xy_step, n_theta, max_range, root=cache_root)
    if os.path.exists(path):
        if verbose:
            print(f'[grid] loading cache {path}')
        return LidarGrid.load(path)
    grid = build_grid_for_env(env_name, lidar_config=lidar_config, xy_step=xy_step,
                              n_theta=n_theta, verbose=verbose)
    grid.save(path)
    if verbose:
        print(f'[grid] cached -> {path}')
    return grid
