"""Grid-guided route -> LiDAR-scan-plan synthesis (the PM likelihood-grid navigator).

The missing middle of the closed loop: localization (LiveLocalizer / GridLocalizer)
tells the ant WHERE it is on the (x, y, theta) grid and where the GOAL cell is; this
module turns those two cells into an executable *scan plan* -- the same object the
diffusion planner outputs -- so the existing waypoint-follower + inverse-dynamics
executor consume it unchanged:

    BFS route over free cells -> world polyline (light corner smoothing, verified to
    stay in free space) -> resample at dataset speed (~0.033 cell/frame) -> heading
    profile along the path (circular-safe smoothing + optional ramp from the ant's
    current estimated yaw) -> ray-cast a scan at every pose (scanner.scan_batch).

Everything is pure numpy on top of :class:`LidarScanner`; no torch / mujoco.
Compliance is the repo standard: (x, y) is used only to *render* scans (exactly like
the goal scan and the reference grid); the executor still sees only scans.

Also provides:
  * ``probe``            -- cold-start fallback: a short scan plan straight down the
                            most-open corridor (movement creates the motion parallax
                            the histogram filter needs to converge).
  * ``band_along_route`` -- a goal "band" (K strided scans) rendered along the real
                            approach corridor of the route, replacing the straight-
                            line most-open-beam eval synthesis (train/eval OOD fix).
"""

from collections import deque

import numpy as np


# ---------------------------------------------------------------------------#
# BFS over the maze occupancy grid
# ---------------------------------------------------------------------------#
def bfs_cell_path(maze_map, start_cell, goal_cell):
    """Shortest 4-connected path over free cells, inclusive of both endpoints.

    Returns an (N, 2) int array of (i, j) cells, or ``None`` if either endpoint is
    a wall / out of bounds or no path exists.
    """
    mp = np.asarray(maze_map)
    H, W = mp.shape
    si, sj = int(start_cell[0]), int(start_cell[1])
    gi, gj = int(goal_cell[0]), int(goal_cell[1])
    for (i, j) in ((si, sj), (gi, gj)):
        if not (0 <= i < H and 0 <= j < W) or mp[i, j] == 1:
            return None
    if (si, sj) == (gi, gj):
        return np.array([[si, sj]], dtype=np.int64)
    parent = -np.ones((H, W, 2), dtype=np.int32)
    seen = np.zeros((H, W), dtype=bool)
    seen[si, sj] = True
    q = deque([(si, sj)])
    while q:
        i, j = q.popleft()
        if (i, j) == (gi, gj):
            break
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W and mp[ni, nj] == 0 and not seen[ni, nj]:
                seen[ni, nj] = True
                parent[ni, nj] = (i, j)
                q.append((ni, nj))
    if not seen[gi, gj]:
        return None
    path = [(gi, gj)]
    while path[-1] != (si, sj):
        i, j = path[-1]
        pi, pj = parent[i, j]
        path.append((int(pi), int(pj)))
    return np.asarray(path[::-1], dtype=np.int64)


# ---------------------------------------------------------------------------#
# angle helpers (circular-safe)
# ---------------------------------------------------------------------------#
def _smooth_headings(yaw, win):
    """Moving-average smoothing of headings via unit vectors (no wrap artifacts)."""
    if win <= 1 or len(yaw) < 3:
        return yaw
    v = np.stack([np.cos(yaw), np.sin(yaw)], axis=1)
    k = np.ones(int(win)) / float(win)
    pad = int(win) // 2
    vp = np.pad(v, ((pad, pad), (0, 0)), mode='edge')
    vs = np.stack([np.convolve(vp[:, 0], k, mode='valid'),
                   np.convolve(vp[:, 1], k, mode='valid')], axis=1)
    vs = vs[:len(yaw)]
    return np.arctan2(vs[:, 1], vs[:, 0])


def _ang_lerp(a, b, w):
    """Geodesic blend from angle ``a`` to ``b`` with weight ``w`` in [0, 1]."""
    d = np.arctan2(np.sin(b - a), np.cos(b - a))
    return a + w * d


# ---------------------------------------------------------------------------#
# RouteScanSynthesizer
# ---------------------------------------------------------------------------#
class RouteScanSynthesizer:
    """Turn (start pose estimate, goal point) into a dense, executable scan plan.

    Args:
        scanner: a :class:`LidarScanner` (the SAME geometry that built the training
            data and the reference grid).
        step_wu: world units between consecutive plan frames. Dataset frames are
            ~0.033 cell = ~0.13 wu apart, so the default matches the speed the
            inverse-dynamics model was trained at.
        yaw_smooth_win: heading smoothing window, in frames (~31 frames = 1 cell).
        smooth_path: 1 round of Chaikin corner-cutting on the cell-center polyline
            (kept only if every resampled point stays in free space).
    """

    def __init__(self, scanner, step_wu=0.13, yaw_smooth_win=31, smooth_path=True):
        self.scanner = scanner
        self.step_wu = float(step_wu)
        self.yaw_smooth_win = int(yaw_smooth_win)
        self.smooth_path = bool(smooth_path)

    # -- geometry -----------------------------------------------------------#
    def cell_center(self, i, j):
        sc = self.scanner
        return (float(j) * sc.maze_unit - sc.offset_x,
                float(i) * sc.maze_unit - sc.offset_y)

    def _points_free(self, pts):
        i, j = self.scanner.world_to_cell(pts[:, 0], pts[:, 1])
        return not np.any(self.scanner._is_wall(i, j))

    def _polyline(self, cells, start_xy=None, end_xy=None):
        pts = np.array([self.cell_center(i, j) for i, j in cells], dtype=np.float64)
        if start_xy is not None:
            pts = np.concatenate([np.asarray(start_xy, dtype=np.float64)[None], pts], 0)
        if end_xy is not None:
            pts = np.concatenate([pts, np.asarray(end_xy, dtype=np.float64)[None]], 0)
        ## drop consecutive duplicates (e.g. start_xy == first cell center)
        keep = np.ones(len(pts), dtype=bool)
        keep[1:] = np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-6
        return pts[keep]

    @staticmethod
    def _chaikin(pts):
        """One round of corner cutting; endpoints preserved."""
        if len(pts) < 3:
            return pts
        out = [pts[0]]
        for a, b in zip(pts[:-1], pts[1:]):
            out.append(0.75 * a + 0.25 * b)
            out.append(0.25 * a + 0.75 * b)
        out.append(pts[-1])
        return np.asarray(out)

    def _resample(self, pts):
        """Resample a polyline at ``step_wu`` spacing (arclength-uniform)."""
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(s[-1])
        if total < 1e-9:
            return pts[:1].copy()
        n = max(int(np.floor(total / self.step_wu)) + 1, 2)
        si = np.linspace(0.0, total, n)
        x = np.interp(si, s, pts[:, 0])
        y = np.interp(si, s, pts[:, 1])
        return np.stack([x, y], axis=1)

    # -- main synthesis -----------------------------------------------------#
    def synthesize(self, start_xy, goal_xy, start_yaw=None, goal_hold=0,
                   yaw_ramp_frames=45):
        """(start point, goal point) -> executable scan plan.

        Returns dict(scans (T, nb) float32, xy (T, 2), yaw (T,), cells (N, 2))
        or ``None`` when no route exists (caller should fall back to ``probe``).
        ``goal_hold`` appends that many copies of the final pose (park at goal so
        the env success check / arrival detector latch while the executor holds).
        """
        sc = self.scanner
        s_cell = sc.world_to_cell(np.asarray([start_xy[0]]), np.asarray([start_xy[1]]))
        g_cell = sc.world_to_cell(np.asarray([goal_xy[0]]), np.asarray([goal_xy[1]]))
        cells = bfs_cell_path(sc.maze_map, (int(s_cell[0][0]), int(s_cell[1][0])),
                              (int(g_cell[0][0]), int(g_cell[1][0])))
        if cells is None:
            return None
        poly = self._polyline(cells, start_xy=start_xy, end_xy=goal_xy)
        if self.smooth_path and len(poly) >= 3:
            sm = self._chaikin(poly)
            rs = self._resample(sm)
            if self._points_free(rs):
                poly, pts = sm, rs
            else:                                   ## smoothing clipped a wall corner
                pts = self._resample(poly)
        else:
            pts = self._resample(poly)

        ## heading = smoothed local tangent
        if len(pts) >= 2:
            d = np.gradient(pts, axis=0)
            yaw = np.arctan2(d[:, 1], d[:, 0])
            yaw = _smooth_headings(yaw, self.yaw_smooth_win)
        else:
            yaw = np.array([0.0 if start_yaw is None else float(start_yaw)])
        ## ramp from the ant's current heading into the path heading
        if start_yaw is not None and len(yaw) > 1:
            r = min(int(yaw_ramp_frames), len(yaw))
            w = np.linspace(0.0, 1.0, r)
            yaw[:r] = _ang_lerp(float(start_yaw) * np.ones(r), yaw[:r], w)

        if goal_hold > 0:
            pts = np.concatenate([pts, np.repeat(pts[-1:], goal_hold, axis=0)], 0)
            yaw = np.concatenate([yaw, np.repeat(yaw[-1:], goal_hold)], 0)

        scans = sc.scan_batch(pts, yaw).astype(np.float32)
        return dict(scans=scans, xy=pts.astype(np.float32),
                    yaw=yaw.astype(np.float32), cells=cells)

    # -- cold-start probe ---------------------------------------------------#
    def probe(self, est_xy, est_yaw, current_scan, length_wu=8.0, goal_hold=0):
        """Short plan straight down the most-open corridor of the CURRENT scan.

        Used when the position belief is still too uncertain for a BFS route (the
        open direction is body-relative, so it is valid even if ``est_xy`` is off;
        moving creates the parallax the filter needs). Scans are rendered from the
        current *estimated* pose -- they self-correct at the next replan.
        """
        sc = self.scanner
        scan = np.asarray(current_scan, dtype=np.float32)
        b = int(np.argmax(scan))
        ang = float(est_yaw) + float(sc.beam_offsets[b])
        L = float(min(max(float(scan[b]) - 1.0, 1.0), length_wu))
        n = max(int(L / self.step_wu), 2)
        t = np.linspace(0.0, L, n)
        pts = np.stack([est_xy[0] + t * np.cos(ang),
                        est_xy[1] + t * np.sin(ang)], axis=1)
        ## clip to free space (stop before the first wall point, keep >= 2 frames)
        i, j = sc.world_to_cell(pts[:, 0], pts[:, 1])
        bad = np.where(sc._is_wall(i, j))[0]
        if len(bad) > 0:
            pts = pts[:max(int(bad[0]), 2)]
        yaw = np.full(len(pts), ang)
        r = min(30, len(pts))
        yaw[:r] = _ang_lerp(float(est_yaw) * np.ones(r), yaw[:r], np.linspace(0, 1, r))
        if goal_hold > 0:
            pts = np.concatenate([pts, np.repeat(pts[-1:], goal_hold, axis=0)], 0)
            yaw = np.concatenate([yaw, np.repeat(yaw[-1:], goal_hold)], 0)
        scans = sc.scan_batch(pts, yaw).astype(np.float32)
        return dict(scans=scans, xy=pts.astype(np.float32),
                    yaw=yaw.astype(np.float32), cells=None)

    # -- goal band along the real approach corridor --------------------------#
    def band_along_route(self, route, k, band_step_wu):
        """Last-K strided scans along the route INTO its endpoint -> (K, nb).

        The training goal band is K real strided frames of the actual (curved,
        heading-varying) approach; this renders exactly that from the BFS route,
        replacing the straight-line most-open-beam synthesis. ``route`` is the
        dict returned by :meth:`synthesize`.
        """
        xy, yaw = route['xy'], route['yaw']
        stride = max(int(round(float(band_step_wu) / self.step_wu)), 1)
        idx = [len(xy) - 1 - j * stride for j in range(int(k))]
        idx = np.clip(np.asarray(idx[::-1], dtype=int), 0, len(xy) - 1)
        scans = self.scanner.scan_batch(xy[idx].astype(np.float64),
                                        yaw[idx].astype(np.float64))
        return scans.astype(np.float32)
