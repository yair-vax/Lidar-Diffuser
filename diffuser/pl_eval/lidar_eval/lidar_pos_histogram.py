"""
Position-histogram evaluation for the LiDAR antmaze model (project spec, stage 6).

What it produces (all overlaid on the maze layout, saved as .png):

  1. Test-set position histogram
        Where, in the maze, do the held-out (test) trajectories spend their time?
        A 2D occupancy heatmap of agent (x, y) over the test split.

  2. Per-cell LiDAR ambiguity map  (structure-only difficulty predictor)
        For every free maze cell we average the LiDAR scans observed there and
        measure how similar that signature is to *other* cells. Cells whose LiDAR
        signature is easily confused with others (open rooms, symmetric corridors)
        are intrinsically harder to localize / reconstruct from LiDAR alone. This
        is the "correlation between maze structure and model difficulty" the spec
        asks to surface.

  3. (optional) Reconstruction-error map
        If a model output file is supplied via --recon_npz (keys: ``pred_lidar``
        and ``true_lidar``, or ``pred_xy`` and ``true_xy``), the per-cell mean
        reconstruction error is binned onto the maze, and its correlation with the
        ambiguity map (2) is reported -- i.e. "do the structurally-ambiguous
        locations coincide with where the model performs worst?".

Runs with numpy + matplotlib only (no torch / OGBench). Input data can be either
a LiDAR npz produced by ``gen_lidar_dataset.py`` or a raw OGBench npz (LiDAR is
then computed on the fly).

Example::

    python -m diffuser.pl_eval.lidar_eval.lidar_pos_histogram \
        --npz data/ogb_maze/antmaze-giant-stitch-v0-luotest.npz \
        --env antmaze-giant-stitch-v0 \
        --out logs/lidar_eval/giant_luotest
"""

import os
import json
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# --- robust import of the LiDAR simulator (avoids pulling torch via the package
#     __init__ on machines where torch is not installed) -----------------------#
def _load_lidar_sim():
    try:
        from diffuser.datasets.lidar import lidar_sim as ls
        return ls
    except Exception:
        import importlib.util
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.normpath(os.path.join(
            here, '..', '..', 'datasets', 'lidar', 'lidar_sim.py'))
        spec = importlib.util.spec_from_file_location('lidar_sim', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


LS = _load_lidar_sim()


# ---------------------------------------------------------------------------#
# Data loading
# ---------------------------------------------------------------------------#
def load_inputs(npz_path, env_name, lidar_config):
    """Return dict with xy (N,2), yaw (N,), lidar (N,B), terminals (N,), plus the
    scanner. Works for both LiDAR-format npz and raw OGBench npz."""
    d = np.load(npz_path)
    scanner = LS.make_scanner_for_env(env_name, lidar_config)

    if 'lidar' in d and 'xy' in d:
        xy = d['xy'].astype(np.float64)
        yaw = d['yaw'].astype(np.float64) if 'yaw' in d else np.zeros(len(xy))
        lidar = d['lidar'].astype(np.float32)
    else:
        obs = d['observations']
        xy, yaw = LS.obs_to_xy_yaw(obs, has_quat=obs.shape[1] >= 7)
        lidar = scanner.scan_obs(obs, has_quat=obs.shape[1] >= 7)

    terminals = d['terminals'].astype(bool) if 'terminals' in d else \
        np.zeros(len(xy), dtype=bool)
    return dict(xy=xy, yaw=yaw, lidar=lidar, terminals=terminals, scanner=scanner)


def split_episodes(terminals, test_frac=0.1, seed=0):
    """Deterministic train/test split at the episode level. Returns boolean
    masks (over transitions) for train and test."""
    n = len(terminals)
    ends = np.where(terminals)[0]
    # episode boundaries (start, end_exclusive)
    bounds = []
    prev = 0
    for e in ends:
        bounds.append((prev, e + 1))
        prev = e + 1
    if prev < n:
        bounds.append((prev, n))
    n_ep = len(bounds)

    rng = np.random.default_rng(seed)
    order = rng.permutation(n_ep)
    n_test = max(1, int(round(n_ep * test_frac)))
    test_eps = set(order[:n_test].tolist())

    train_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    for ep, (s, e) in enumerate(bounds):
        (test_mask if ep in test_eps else train_mask)[s:e] = True
    return train_mask, test_mask, n_ep, n_test


# ---------------------------------------------------------------------------#
# Maze drawing helpers
# ---------------------------------------------------------------------------#
def maze_world_extent(scanner):
    """World-coordinate bounding box [xmin, xmax, ymin, ymax] of the whole maze."""
    H, W = scanner.maze_map.shape
    u, ox, oy = scanner.maze_unit, scanner.offset_x, scanner.offset_y
    xmin = 0 * u - ox - 0.5 * u
    xmax = (W - 1) * u - ox + 0.5 * u
    ymin = 0 * u - oy - 0.5 * u
    ymax = (H - 1) * u - oy + 0.5 * u
    return xmin, xmax, ymin, ymax


def draw_walls(ax, scanner):
    H, W = scanner.maze_map.shape
    u, ox, oy = scanner.maze_unit, scanner.offset_x, scanner.offset_y
    for i in range(H):
        for j in range(W):
            if scanner.maze_map[i, j] == 1:
                cx = j * u - ox
                cy = i * u - oy
                ax.add_patch(Rectangle((cx - 0.5 * u, cy - 0.5 * u), u, u,
                                       facecolor='0.25', edgecolor='none', zorder=3))


def cell_centers(scanner):
    """Return free-cell list and a (H,W) world-center grid."""
    H, W = scanner.maze_map.shape
    u, ox, oy = scanner.maze_unit, scanner.offset_x, scanner.offset_y
    free = [(i, j) for i in range(H) for j in range(W) if scanner.maze_map[i, j] == 0]
    return free


# ---------------------------------------------------------------------------#
# Plots
# ---------------------------------------------------------------------------#
def plot_position_histogram(xy, scanner, out_png, title, bins_per_cell=4):
    H, W = scanner.maze_map.shape
    xmin, xmax, ymin, ymax = maze_world_extent(scanner)
    nx = W * bins_per_cell
    ny = H * bins_per_cell
    hist, xedges, yedges = np.histogram2d(
        xy[:, 0], xy[:, 1], bins=[nx, ny], range=[[xmin, xmax], [ymin, ymax]])

    fig, ax = plt.subplots(figsize=(0.6 * W, 0.6 * H))
    # hist is (nx, ny); imshow wants (rows=y, cols=x) -> transpose
    im = ax.imshow(np.log1p(hist.T), origin='lower', extent=[xmin, xmax, ymin, ymax],
                   cmap='magma', aspect='equal', zorder=1)
    draw_walls(ax, scanner)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_title(title)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    fig.colorbar(im, ax=ax, fraction=0.035, label='log(1 + visits)')
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)
    return hist


def per_cell_lidar_signature(xy, lidar, scanner):
    """Mean LiDAR scan per free cell. Returns (cells, sig, counts) where cells is
    a list of (i, j), sig is (n_cells, B), counts is (n_cells,)."""
    i, j = scanner.world_to_cell(xy[:, 0], xy[:, 1])
    H, W = scanner.maze_map.shape
    free = cell_centers(scanner)
    idx_of = {c: k for k, c in enumerate(free)}
    B = lidar.shape[1]
    sig = np.zeros((len(free), B), dtype=np.float64)
    cnt = np.zeros(len(free), dtype=np.int64)
    for n in range(len(xy)):
        c = (int(i[n]), int(j[n]))
        k = idx_of.get(c)
        if k is None:
            continue
        sig[k] += lidar[n]
        cnt[k] += 1
    valid = cnt > 0
    sig[valid] /= cnt[valid, None]
    return free, sig, cnt


def lidar_ambiguity(free, sig, cnt, k_near=3):
    """For each visited cell, ambiguity = mean L2 distance of its mean LiDAR
    signature to its ``k_near`` nearest *other* cells' signatures (smaller =>
    more confusable => harder to localize). Returns (ambig array, valid mask)."""
    valid = cnt > 0
    idxs = np.where(valid)[0]
    S = sig[idxs]
    # pairwise distances among valid cells
    d2 = ((S[:, None, :] - S[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    dist = np.sqrt(np.sort(d2, axis=1)[:, :k_near]).mean(1)
    ambig = np.full(len(free), np.nan)
    ambig[idxs] = dist
    return ambig, valid


def plot_cell_heatmap(values, free, scanner, out_png, title, cmap='viridis',
                      label='value'):
    H, W = scanner.maze_map.shape
    # maze_world_extent spans exactly cell-edge to cell-edge, so a (H, W) image
    # drawn with this extent puts cell (i, j) on its world center (j*u-ox, i*u-oy).
    xmin, xmax, ymin, ymax = maze_world_extent(scanner)
    grid = np.full((H, W), np.nan)
    for (i, j), v in zip(free, values):
        grid[i, j] = v

    fig, ax = plt.subplots(figsize=(0.6 * W, 0.6 * H))
    im = ax.imshow(grid, origin='lower', extent=[xmin, xmax, ymin, ymax],
                   cmap=cmap, aspect='equal', zorder=1, interpolation='nearest')
    draw_walls(ax, scanner)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_title(title); ax.set_xlabel('x'); ax.set_ylabel('y')
    fig.colorbar(im, ax=ax, fraction=0.035, label=label)
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)


def reconstruction_error_per_cell(recon_npz, scanner):
    """Load a model output file and bin its per-transition reconstruction error
    onto maze cells. Supports LiDAR-space error (pred_lidar vs true_lidar) or
    position-space error (pred_xy vs true_xy). Requires ``true_xy`` for binning."""
    d = np.load(recon_npz)
    if 'true_xy' not in d:
        raise ValueError('recon_npz must contain "true_xy" to localize errors.')
    true_xy = d['true_xy'].astype(np.float64)
    if 'pred_lidar' in d and 'true_lidar' in d:
        err = np.sqrt(((d['pred_lidar'].astype(np.float64)
                        - d['true_lidar'].astype(np.float64)) ** 2).mean(1))
        err_label = 'LiDAR RMSE'
    elif 'pred_xy' in d:
        err = np.sqrt(((d['pred_xy'].astype(np.float64) - true_xy) ** 2).sum(1))
        err_label = 'position L2 error'
    else:
        raise ValueError('recon_npz needs (pred_lidar,true_lidar) or pred_xy.')

    i, j = scanner.world_to_cell(true_xy[:, 0], true_xy[:, 1])
    free = cell_centers(scanner)
    idx_of = {c: k for k, c in enumerate(free)}
    acc = np.zeros(len(free)); cnt = np.zeros(len(free))
    for n in range(len(true_xy)):
        k = idx_of.get((int(i[n]), int(j[n])))
        if k is None:
            continue
        acc[k] += err[n]; cnt[k] += 1
    mean_err = np.full(len(free), np.nan)
    mean_err[cnt > 0] = acc[cnt > 0] / cnt[cnt > 0]
    return free, mean_err, err_label


# ---------------------------------------------------------------------------#
# Main
# ---------------------------------------------------------------------------#
def main():
    p = argparse.ArgumentParser(description='LiDAR position-histogram evaluation.')
    p.add_argument('--npz', required=True, help='LiDAR or raw OGBench .npz')
    p.add_argument('--env', required=True, help='OGBench env name')
    p.add_argument('--out', required=True, help='output directory')
    p.add_argument('--n_beams', type=int, default=30)
    p.add_argument('--max_range', type=float, default=12.0)
    p.add_argument('--test_frac', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--bins_per_cell', type=int, default=4)
    p.add_argument('--recon_npz', default=None,
                   help='optional model output (true_xy + pred_lidar/true_lidar or pred_xy)')
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    lidar_config = dict(n_beams=args.n_beams, max_range=args.max_range)
    data = load_inputs(args.npz, args.env, lidar_config)
    scanner = data['scanner']

    train_mask, test_mask, n_ep, n_test = split_episodes(
        data['terminals'], test_frac=args.test_frac, seed=args.seed)
    print(f'[eval] episodes total={n_ep}, test={n_test} '
          f'(test transitions={int(test_mask.sum())})')

    summary = dict(env=args.env, n_episodes=int(n_ep), n_test_episodes=int(n_test),
                   n_test_transitions=int(test_mask.sum()),
                   n_beams=args.n_beams, max_range=args.max_range)

    # 1. Test-set position histogram
    test_xy = data['xy'][test_mask]
    hist_png = os.path.join(args.out, 'test_position_histogram.png')
    plot_position_histogram(test_xy, scanner, hist_png,
                            f'{args.env}: test-set position histogram',
                            bins_per_cell=args.bins_per_cell)
    print(f'[eval] wrote {hist_png}')

    # 2. Per-cell LiDAR ambiguity (structure-driven difficulty), on test split
    free, sig, cnt = per_cell_lidar_signature(test_xy, data['lidar'][test_mask], scanner)
    ambig, valid = lidar_ambiguity(free, sig, cnt)
    amb_png = os.path.join(args.out, 'lidar_ambiguity_map.png')
    plot_cell_heatmap(ambig, free, scanner, amb_png,
                      f'{args.env}: LiDAR ambiguity (low = harder to localize)',
                      cmap='viridis_r', label='mean dist to nearest cells')
    print(f'[eval] wrote {amb_png}')
    summary['mean_lidar_ambiguity'] = float(np.nanmean(ambig))
    summary['n_visited_cells'] = int(valid.sum())

    # 3. Optional reconstruction-error map + correlation with ambiguity
    if args.recon_npz is not None:
        free_r, mean_err, err_label = reconstruction_error_per_cell(args.recon_npz, scanner)
        err_png = os.path.join(args.out, 'reconstruction_error_map.png')
        plot_cell_heatmap(mean_err, free_r, scanner, err_png,
                          f'{args.env}: reconstruction error per cell',
                          cmap='magma', label=err_label)
        print(f'[eval] wrote {err_png}')
        # correlation: harder (low ambiguity) cells should have higher error
        both = (~np.isnan(ambig)) & (~np.isnan(np.array(mean_err)))
        if both.sum() >= 3:
            corr = float(np.corrcoef(ambig[both], np.array(mean_err)[both])[0, 1])
            summary['corr_ambiguity_vs_error'] = corr
            print(f'[eval] corr(ambiguity, error) = {corr:.3f} '
                  f'(negative => ambiguous cells reconstruct worse)')
            fig, ax = plt.subplots(figsize=(4.5, 4))
            ax.scatter(ambig[both], np.array(mean_err)[both], s=14, alpha=0.7)
            ax.set_xlabel('LiDAR ambiguity (higher = more distinct)')
            ax.set_ylabel(err_label)
            ax.set_title(f'structure vs performance (r={corr:.2f})')
            fig.tight_layout()
            fig.savefig(os.path.join(args.out, 'ambiguity_vs_error_scatter.png'), dpi=120)
            plt.close(fig)

    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'[eval] summary: {summary}')


if __name__ == '__main__':
    main()
