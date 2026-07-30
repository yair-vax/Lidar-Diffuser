#!/usr/bin/env python
"""Sequence (motion-history) vs single-scan localization ambiguity  (no models, no MuJoCo).

Decides whether to fund a HISTORY-CONDITIONED planner. Every fix so far tried to pin
position from ONE LiDAR frame (goal-side band/range, plan-side goal-pick) and hit the
maze's self-similarity. The one signal we never used is TIME: the agent's *sequence* of
scans as it moves. A single scan may be ambiguous while the last K scans of an approach
are nearly unique. This script measures exactly that, on REAL val trajectories.

For a sample of trajectory anchor frames it counts, per anchor, how many OTHER far-away
anchors (>= min_cell_gap cells) have a matching observation, where "observation" is:
  K=1  -> just the current scan          (the single-scan baseline = today's setting)
  K>1  -> the last K strided scans       (a motion history ending at the anchor)
matched by mean per-beam RMS over the K frames.

READ:
  * %-ambiguous / mean look-alikes DROP sharply as K grows
        -> motion history disambiguates position -> fund a history-conditioned planner.
  * stay flat
        -> even a sequence is ambiguous here -> history won't help; rethink the task.

Run (laptop or server; pure numpy):
    python diffuser/pl_eval/lidar_eval/seq_ambiguity.py \
        --env antmaze-giant-stitch-v0 \
        --npz ~/.ogbench/data/antmaze-giant-stitch-v0-val.npz \
        --ks 1 2 4 8 --hist_stride 15 --out logs/lidar_eval/seq_ambiguity.json
"""
import argparse, json, os, importlib.util
import numpy as np


def _load_make_scanner():
    try:
        from diffuser.datasets.lidar.lidar_sim import make_scanner_for_env
        return make_scanner_for_env
    except Exception:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.normpath(os.path.join(here, '..', '..', 'datasets', 'lidar', 'lidar_sim.py'))
        spec = importlib.util.spec_from_file_location('lidar_sim', path)
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        return mod.make_scanner_for_env


make_scanner_for_env = _load_make_scanner()


def load_val(npz_path):
    d = np.load(os.path.expanduser(npz_path), allow_pickle=True)
    obs = np.asarray(d['observations'], dtype=np.float32)
    term = None
    for k in ('terminals', 'dones', 'timeouts'):
        if k in d.files:
            term = np.asarray(d[k]).reshape(-1).astype(bool); break
    if term is not None and term.shape[0] == obs.shape[0]:
        ep_id = np.cumsum(np.concatenate([[0], term[:-1].astype(np.int64)]))
    else:
        ep_id = np.zeros(obs.shape[0], dtype=np.int64)
    return obs, ep_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', default='antmaze-giant-stitch-v0')
    ap.add_argument('--npz', default='~/.ogbench/data/antmaze-giant-stitch-v0-val.npz')
    ap.add_argument('--n_beams', type=int, default=30)
    ap.add_argument('--max_range', type=float, default=12.0)
    ap.add_argument('--ks', type=int, nargs='+', default=[1, 2, 4, 8],
                    help='history lengths to compare (K=1 == single scan baseline)')
    ap.add_argument('--hist_stride', type=int, default=15,
                    help='frames between history scans (~0.5 cell at mean|dxy|=0.132 wu)')
    ap.add_argument('--n_anchors', type=int, default=800)
    ap.add_argument('--min_cell_gap', type=float, default=3.0)
    ap.add_argument('--tols', type=float, nargs='+', default=[0.25, 0.5, 1.0])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='logs/lidar_eval/seq_ambiguity.json')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    scanner = make_scanner_for_env(
        args.env, dict(n_beams=args.n_beams, max_range=args.max_range, fov=2.0 * np.pi))
    obs, ep_id = load_val(args.npz)
    mu = scanner.maze_unit
    K_max, S = max(args.ks), args.hist_stride
    need = (K_max - 1) * S

    ## anchors = frames with a full strided in-episode history for the largest K
    elig = np.where((np.arange(len(obs)) - need >= 0))[0]
    elig = elig[ep_id[elig - need] == ep_id[elig]]
    if len(elig) == 0:
        raise RuntimeError(f'no frames with {need} in-episode predecessors; lower --ks/--hist_stride')
    anchors = rng.choice(elig, size=min(args.n_anchors, len(elig)), replace=False)
    M = len(anchors)

    ## render the K_max-frame strided history scans for every anchor (newest -> oldest)
    hist_idx = anchors[:, None] - np.arange(K_max)[None, :] * S      # (M, K_max)
    flat = hist_idx.reshape(-1)
    scans_flat = scanner.scan_obs(obs[flat], has_quat=True)          # (M*K_max, n_beams)
    hist = scans_flat.reshape(M, K_max, args.n_beams).astype(np.float32)

    ## pairwise cell-distance between anchors; exclude near neighbours + self
    axy = obs[anchors, :2].astype(np.float64)
    cd = np.sqrt(((axy[:, None, :] - axy[None, :, :]) ** 2).sum(-1)) / mu   # (M,M) in cells
    far = cd >= args.min_cell_gap

    report = {'env': args.env, 'n_anchors': int(M), 'hist_stride': S,
              'min_cell_gap': args.min_cell_gap, 'tols': args.tols, 'ks': {}}
    print(f'env={args.env}  anchors={M}  hist_stride={S} (~{S*0.132/mu:.2f} cell/scan)')
    for K in args.ks:
        sub = hist[:, :K, :]                                         # newest K frames
        ## mean per-beam RMS between two K-frame histories (aligned newest->oldest)
        seqdist = np.empty((M, M), dtype=np.float32)
        CH = 48
        for a in range(0, M, CH):
            d = sub[a:a + CH, None, :, :] - sub[None, :, :, :]       # (<=CH,M,K,beams)
            seqdist[a:a + CH] = np.sqrt((d ** 2).mean(-1)).mean(-1)  # mean over beams then K
        per_tol = {}
        for tol in args.tols:
            look = far & (seqdist < tol)
            nl = look.sum(1)
            per_tol[f'{tol}'] = dict(frac_ambiguous=float((nl > 0).mean()),
                                     mean_lookalikes=float(nl.mean()),
                                     median_lookalikes=float(np.median(nl)))
        report['ks'][f'{K}'] = dict(span_cells=float((K - 1) * S * 0.132 / mu), per_tol=per_tol)
        line = '  '.join(f'tol{t}: amb {per_tol[f"{t}"]["frac_ambiguous"]*100:4.1f}% '
                         f'(mean {per_tol[f"{t}"]["mean_lookalikes"]:4.1f})' for t in args.tols)
        print(f'  K={K:2d} (~{(K-1)*S*0.132/mu:.1f} cell span):  {line}')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'saved -> {args.out}')
    print('READ: if %ambiguous collapses from K=1 to K>1, motion history localizes the '
          'agent -> fund a history-conditioned planner; if flat, it will not help.')


if __name__ == '__main__':
    main()
