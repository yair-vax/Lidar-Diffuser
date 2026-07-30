"""Offline unit test of _route_to_data_ribbon (A2) with a mock scanner.

Mirrors the method added to ogb_stgl_sml_lidar_planner_v1.py 2026-07-07.
Scenarios:
  1. dense corridor, both headings present -> picks heading-aligned frames,
     covers the route, small end-gap, hold appended, no synth fill
  2. sparse stretch (no frames in the middle) -> synth fill kicks in
  3. wrong-heading-only data -> relaxed pass still builds a ribbon
"""
import numpy as np


class MockScanner:
    maze_unit = 4.0
    offset_x = 4.0
    offset_y = 4.0
    n_beams = 30

    def world_to_cell(self, x, y):
        x = np.asarray(x); y = np.asarray(y)
        j = np.floor((x + self.offset_x) / self.maze_unit).astype(int)
        i = np.floor((y + self.offset_y) / self.maze_unit).astype(int)
        return i, j

    def scan_obs(self, obs, has_quat=True):
        out = np.zeros((len(obs), self.n_beams), dtype=np.float32)
        out[:, 0] = obs[:, 0]          # tag with x so ordering is checkable
        return out

    def scan_batch(self, pts, yaws):
        out = np.full((len(pts), self.n_beams), -1.0, dtype=np.float32)  # fill marker
        return out


def obs_to_xy_yaw(obs, has_quat=True):
    return obs[:, :2], obs[:, 2]       # mock: yaw stored in col 2


class Utils:
    @staticmethod
    def print_color(s, c=None):
        print(s)


utils = Utils()


class A2Host:
    """Host with just the attrs _route_to_data_ribbon touches."""
    def __init__(self, data, seg=12, snap=2.0, max_dyaw=60.0, hold=5):
        self.scanner = MockScanner()
        self.n_beams = 30
        self._data = data
        self._nav_a2_seg = seg
        self._nav_a2_snap_wu = snap
        self._nav_goal_hold = hold

        class Args:
            nav_tf_max_dyaw = max_dyaw
        self.args = Args()

    def _load_tf_data(self):
        return self._data

    # ---- verbatim logic from the planner method (imports adapted) ----
    def _route_to_data_ribbon(self, route):
        data = self._load_tf_data()
        ep_id = data['ep_id']
        if 'ep_end' not in data:
            N = len(ep_id)
            ep_end = np.empty(N, dtype=np.int64)
            last = N - 1
            for i in range(N - 1, -1, -1):
                if i < N - 1 and ep_id[i] != ep_id[i + 1]:
                    last = i
                ep_end[i] = last
            data['ep_end'] = ep_end
        ep_end = data['ep_end']
        if 'yaw' not in data:
            data['yaw'] = obs_to_xy_yaw(data['obs'], has_quat=True)[1].astype(np.float32)
        dyaw_all = data['yaw']
        dxy = data['xy']
        if 'cell_buckets' not in data:
            ci, cj = self.scanner.world_to_cell(dxy[:, 0], dxy[:, 1])
            buck = {}
            for n_i, key in enumerate(zip(np.asarray(ci).tolist(), np.asarray(cj).tolist())):
                buck.setdefault(key, []).append(n_i)
            data['cell_buckets'] = {k: np.asarray(v, dtype=np.int64) for k, v in buck.items()}
        buckets = data['cell_buckets']

        def _cands(pt, k_keep=64):
            pi, pj = self.scanner.world_to_cell(np.asarray([pt[0]]), np.asarray([pt[1]]))
            pools = [buckets[key] for di in (-1, 0, 1) for dj in (-1, 0, 1)
                     if (key := (int(pi[0]) + di, int(pj[0]) + dj)) in buckets]
            if not pools:
                return np.empty(0, np.int64), np.empty(0, np.float64)
            idxs = np.concatenate(pools)
            d = np.linalg.norm(dxy[idxs].astype(np.float64) - pt[None, :], axis=1)
            o = np.argsort(d)[:k_keep]
            return idxs[o], d[o]

        rxy = np.asarray(route['xy'], dtype=np.float64)
        ryaw = np.asarray(route['yaw'], dtype=np.float64)
        n_r = len(rxy)
        while n_r > 1 and float(np.linalg.norm(rxy[n_r - 1] - rxy[n_r - 2])) < 1e-9:
            n_r -= 1
        max_dyaw = float(getattr(self.args, 'nav_tf_max_dyaw', 60.0)) * np.pi / 180.0
        seg_len = max(int(self._nav_a2_seg), 1)
        snap_wu = float(self._nav_a2_snap_wu)

        rows = []
        fill_xy, fill_yaw = [], []
        p, n_seg, n_relax, n_fill = 0, 0, 0, 0
        snap_ds = []
        guard = 0
        while p < n_r - 1 and guard < 4 * n_r + 64:
            guard += 1
            ci, cd = _cands(rxy[p])
            pick = -1
            if len(ci) > 0:
                dy = np.abs(np.arctan2(np.sin(dyaw_all[ci] - ryaw[p]),
                                       np.cos(dyaw_all[ci] - ryaw[p])))
                for pass_dyaw in (max_dyaw, np.inf):
                    ok = np.where((cd <= snap_wu) & (dy <= pass_dyaw))[0]
                    if len(ok) > 0:
                        pick = int(ci[ok[0]])
                        snap_ds.append(float(cd[ok[0]]))
                        n_relax += int(pass_dyaw is np.inf)
                        break
            if pick < 0:
                q = min(p + seg_len, n_r - 1)
                for t in range(p, q):
                    rows.append((False, len(fill_xy)))
                    fill_xy.append(rxy[t]); fill_yaw.append(ryaw[t])
                n_fill += q - p
                p = q
                continue
            seg = np.arange(pick, int(min(pick + seg_len - 1, ep_end[pick])) + 1,
                            dtype=np.int64)
            w = rxy[p:min(p + 4 * seg_len, n_r)]
            dmat = np.linalg.norm(dxy[seg].astype(np.float64)[:, None, :]
                                  - w[None, :, :], axis=2)
            off = np.where(dmat.min(axis=1) > 1.5 * snap_wu)[0]
            if len(off) > 0:
                seg = seg[:max(int(off[0]), 1)]
                dmat = dmat[:len(seg)]
            rows.extend((True, int(v)) for v in seg)
            n_seg += 1
            p += max(int(dmat[-1].argmin()), 1)

        if not rows:
            raise RuntimeError('A2 produced an empty ribbon')
        d_idx = np.asarray([i for is_d, i in rows if is_d], dtype=np.int64)
        scans_d = (self.scanner.scan_obs(data['obs'][d_idx], has_quat=True)
                   .astype(np.float32) if len(d_idx) else
                   np.zeros((0, self.n_beams), dtype=np.float32))
        if fill_xy:
            scans_f = self.scanner.scan_batch(
                np.asarray(fill_xy, dtype=np.float64),
                np.asarray(fill_yaw, dtype=np.float64)).astype(np.float32)
        T = len(rows)
        scans = np.empty((T, scans_d.shape[1] if len(d_idx) else scans_f.shape[1]),
                         dtype=np.float32)
        xy_out = np.empty((T, 2), dtype=np.float32)
        yaw_out = np.empty(T, dtype=np.float32)
        di = 0
        for t, (is_d, i) in enumerate(rows):
            if is_d:
                scans[t] = scans_d[di]; di += 1
                xy_out[t] = dxy[i]; yaw_out[t] = dyaw_all[i]
            else:
                scans[t] = scans_f[i]
                xy_out[t] = fill_xy[i]; yaw_out[t] = fill_yaw[i]
        hold = int(getattr(self, '_nav_goal_hold', 0))
        if hold > 0:
            scans = np.concatenate([scans, np.repeat(scans[-1:], hold, axis=0)], 0)
            xy_out = np.concatenate([xy_out, np.repeat(xy_out[-1:], hold, axis=0)], 0)
            yaw_out = np.concatenate([yaw_out, np.repeat(yaw_out[-1:], hold)], 0)
        sd = np.asarray(snap_ds) if snap_ds else np.asarray([np.nan])
        end_gap = float(np.linalg.norm(xy_out[-1].astype(np.float64) - rxy[n_r - 1]))
        utils.print_color(
            f'[gridnav/A2] ribbon: {T} frames ({len(d_idx)} real / {n_fill} synth-fill), '
            f'{n_seg} segs, snap p50={np.nanmedian(sd):.2f} p90={np.nanpercentile(sd, 90):.2f} wu, '
            f'{n_relax} yaw-relaxed, end-gap {end_gap:.2f} wu')
        return dict(scans=scans, xy=xy_out, yaw=yaw_out, cells=route['cells'])


def make_corridor_data(x0=0.0, x1=40.0, dx=0.132, y=0.0, jitter=0.05, both_ways=True, seed=0):
    rng = np.random.RandomState(seed)
    xs = np.arange(x0, x1, dx)
    n = len(xs)
    fwd = np.stack([xs, y + rng.uniform(-jitter, jitter, n),
                    np.zeros(n) + rng.uniform(-0.2, 0.2, n)], axis=1)  # yaw ~ 0
    eps = [fwd]
    if both_ways:
        bwd = fwd[::-1].copy()
        bwd[:, 2] = np.pi + rng.uniform(-0.2, 0.2, n)                   # yaw ~ pi
        eps.append(bwd)
    obs = np.concatenate(eps, axis=0).astype(np.float32)
    ep_id = np.concatenate([np.full(len(e), k) for k, e in enumerate(eps)]).astype(np.int64)
    return dict(obs=obs, xy=obs[:, :2].astype(np.float32), ep_id=ep_id)


def make_route(x0=0.0, x1=40.0, y=0.0, step=0.13, hold=5):
    xs = np.arange(x0, x1 + 1e-9, step)
    xy = np.stack([xs, np.full(len(xs), y)], axis=1)
    yaw = np.zeros(len(xs))
    xy = np.concatenate([xy, np.repeat(xy[-1:], hold, axis=0)], 0)      # goal_hold tail
    yaw = np.concatenate([yaw, np.repeat(yaw[-1:], hold)], 0)
    return dict(xy=xy.astype(np.float32), yaw=yaw.astype(np.float32),
                scans=np.zeros((len(xy), 30), np.float32), cells=[(0, 0), (0, 1)])


def t1_dense():
    data = make_corridor_data()
    host = A2Host(data)
    route = make_route()
    out = host._route_to_data_ribbon(route)
    n_r = len(route['xy']) - 5
    assert out['scans'].shape[0] == out['xy'].shape[0] == out['yaw'].shape[0]
    ## heading gate: all chosen yaws ~0, not pi
    body = out['yaw'][:-5]
    frac_fwd = np.mean(np.abs(np.arctan2(np.sin(body), np.cos(body))) < np.pi / 2)
    assert frac_fwd > 0.95, frac_fwd
    ## coverage: x must progress monotonically-ish from ~0 to ~40
    x = out['xy'][:, 0].astype(np.float64)
    assert x[0] < 2.0 and x[-1] > 37.5, (x[0], x[-1])
    assert np.mean(np.diff(x[:-5]) >= -0.5) > 0.98            # tiny backtracks ok at seams
    ## no synth fill markers (-1 in beam space)
    assert not np.any(out['scans'][:, 1] == -1.0)
    ## hold: last 5 rows identical
    assert np.allclose(out['xy'][-5:], out['xy'][-1])
    print('t1 dense corridor: PASS\n')


def t2_sparse_gap():
    data = make_corridor_data(x0=0.0, x1=15.0)
    d2 = make_corridor_data(x0=25.0, x1=40.0, seed=1)
    obs = np.concatenate([data['obs'], d2['obs']], 0)
    ep_id = np.concatenate([data['ep_id'], d2['ep_id'] + data['ep_id'].max() + 1], 0)
    data = dict(obs=obs, xy=obs[:, :2].astype(np.float32), ep_id=ep_id)
    host = A2Host(data)
    out = host._route_to_data_ribbon(make_route())
    fills = np.sum(out['scans'][:, 1] == -1.0)
    assert fills > 0, 'expected synth fill across the 15-25 gap'
    x = out['xy'][:, 0].astype(np.float64)
    assert x[-1] > 37.5
    print(f't2 sparse gap: PASS (fills={fills})\n')


def t3_wrong_heading_only():
    data = make_corridor_data(both_ways=False)
    data['obs'][:, 2] = np.pi                                  # everyone walks -x
    data['yaw'] = data['obs'][:, 2].astype(np.float32)
    host = A2Host(data)
    out = host._route_to_data_ribbon(make_route())
    assert out['scans'].shape[0] > 100
    print('t3 wrong-heading relax: PASS\n')


if __name__ == '__main__':
    t1_dense(); t2_sparse_gap(); t3_wrong_heading_only()
    print('ALL A2 RIBBON TESTS PASS')
