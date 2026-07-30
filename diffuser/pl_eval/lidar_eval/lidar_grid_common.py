"""
Shared helpers for the LiDAR (x, y, theta) grid-localization CLIs.

Robust, torch-free imports of the grid + localization library (so every CLI runs
on a bare CPU box) plus the maze-overlay plotting used by all the map scripts.
Plotting convention matches ``lidar_pos_histogram.py`` (world-coordinate extent,
``origin='lower'``, dark wall rectangles).
"""

import os
import importlib.util
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ---------------------------------------------------------------------------#
# torch-free imports of our own library modules
# ---------------------------------------------------------------------------#
def _load_module(mod_name, rel_path):
    """Import ``mod_name`` from the package; fall back to a by-path load so the
    CLIs work without importing the (torch-pulling) ``diffuser`` package."""
    try:
        return __import__(mod_name, fromlist=['*'])
    except Exception:  # noqa: BLE001
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.normpath(os.path.join(here, rel_path))
        spec = importlib.util.spec_from_file_location(os.path.basename(rel_path)[:-3], path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


LG = _load_module('diffuser.datasets.lidar.lidar_grid', '../../datasets/lidar/lidar_grid.py')
LL = _load_module('diffuser.datasets.lidar.lidar_localization', '../../datasets/lidar/lidar_localization.py')
LS = _load_module('diffuser.datasets.lidar.lidar_sim', '../../datasets/lidar/lidar_sim.py')


# ---------------------------------------------------------------------------#
# maze overlay plotting
# ---------------------------------------------------------------------------#
def maze_world_extent(scanner):
    H, W = scanner.maze_map.shape
    u, ox, oy = scanner.maze_unit, scanner.offset_x, scanner.offset_y
    return (-ox - 0.5 * u, (W - 1) * u - ox + 0.5 * u,
            -oy - 0.5 * u, (H - 1) * u - oy + 0.5 * u)


def draw_walls(ax, scanner):
    H, W = scanner.maze_map.shape
    u, ox, oy = scanner.maze_unit, scanner.offset_x, scanner.offset_y
    for i in range(H):
        for j in range(W):
            if scanner.maze_map[i, j] == 1:
                ax.add_patch(Rectangle((j * u - ox - 0.5 * u, i * u - oy - 0.5 * u),
                                       u, u, facecolor='0.25', edgecolor='none', zorder=3))


def grid_img_extent(grid):
    """imshow extent for a (n_row, n_col) grid map, centered on the lattice points."""
    h = grid.xy_step / 2.0
    return (grid.x_vals[0] - h, grid.x_vals[-1] + h,
            grid.y_vals[0] - h, grid.y_vals[-1] + h)


def plot_xy_map(grid, values, out_png, title, cmap='viridis', label='value',
                marks=None, log=False):
    """Render a (n_row, n_col) per-(x, y) map over the maze layout.

    marks: optional list of dicts {xy:(x,y), c:color, m:marker, s:size, label:str}.
    """
    scanner = grid.scanner
    H, W = scanner.maze_map.shape
    v = np.array(values, dtype=np.float64)
    if log:
        v = np.log1p(v - np.nanmin(v))
    fig, ax = plt.subplots(figsize=(0.55 * W, 0.55 * H))
    im = ax.imshow(v, origin='lower', extent=grid_img_extent(grid),
                   cmap=cmap, aspect='equal', zorder=1)
    draw_walls(ax, scanner)
    if marks:
        for mk in marks:
            ax.scatter([mk['xy'][0]], [mk['xy'][1]], c=mk.get('c', 'r'),
                       marker=mk.get('m', '*'), s=mk.get('s', 220),
                       edgecolors='white', linewidths=1.3, zorder=5,
                       label=mk.get('label'))
        if any('label' in mk for mk in marks):
            ax.legend(loc='upper left', fontsize=8, framealpha=0.8)
    xmin, xmax, ymin, ymax = maze_world_extent(scanner)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    fig.colorbar(im, ax=ax, fraction=0.035, label=label)
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)
    return out_png


def plot_trajectory(grid, out_png, title, true_xy=None, est_xy=None, marks=None):
    """Overlay a true and/or estimated (x, y) trajectory on the maze."""
    scanner = grid.scanner
    H, W = scanner.maze_map.shape
    fig, ax = plt.subplots(figsize=(0.55 * W, 0.55 * H))
    draw_walls(ax, scanner)
    if true_xy is not None:
        ax.plot(true_xy[:, 0], true_xy[:, 1], '-o', ms=3, lw=1.5, color='tab:green',
                label='true', zorder=4)
    if est_xy is not None:
        ax.plot(est_xy[:, 0], est_xy[:, 1], '-x', ms=4, lw=1.2, color='tab:red',
                label='estimated', zorder=4)
    if marks:
        for mk in marks:
            ax.scatter([mk['xy'][0]], [mk['xy'][1]], c=mk.get('c', 'k'),
                       marker=mk.get('m', '*'), s=mk.get('s', 200),
                       edgecolors='white', linewidths=1.2, zorder=6, label=mk.get('label'))
    xmin, xmax, ymin, ymax = maze_world_extent(scanner)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_title(title, fontsize=11); ax.legend(loc='upper left', fontsize=8, framealpha=0.8)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)
    return out_png


# ---------------------------------------------------------------------------#
# data loading (raw OGBench npz -> obs + episode ids)
# ---------------------------------------------------------------------------#
def load_obs_npz(npz_path):
    d = np.load(os.path.expanduser(npz_path), allow_pickle=True)
    obs = np.asarray(d['observations'], dtype=np.float32)
    term = None
    for k in ('terminals', 'dones', 'timeouts'):
        if k in d.files:
            term = np.asarray(d[k]).reshape(-1).astype(bool)
            break
    if term is not None and term.shape[0] == obs.shape[0]:
        ep_id = np.cumsum(np.concatenate([[0], term[:-1].astype(np.int64)]))
    else:
        ep_id = np.zeros(obs.shape[0], dtype=np.int64)
    return obs, ep_id


def lidar_config_from_args(args):
    return dict(n_beams=args.n_beams, max_range=args.max_range, fov=2.0 * np.pi)


def add_grid_args(ap):
    """Common grid/localization CLI arguments."""
    ap.add_argument('--env', default='antmaze-giant-stitch-v0')
    ap.add_argument('--n_beams', type=int, default=30)
    ap.add_argument('--max_range', type=float, default=12.0)
    ap.add_argument('--xy_step', type=float, default=1.0,
                    help='grid spatial resolution, world units (1 cell = maze_unit)')
    ap.add_argument('--n_theta', type=int, default=12,
                    help='orientation bins over 360 deg (12 == 30 deg/bin)')
    ap.add_argument('--sigma', type=float, default=1.0,
                    help='measurement-likelihood noise scale (world units)')
    ap.add_argument('--cache_root', default='data/ogb_maze/lidar_grid')
    return ap
