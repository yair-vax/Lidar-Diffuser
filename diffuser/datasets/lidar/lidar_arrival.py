"""PM Stage-3 closed-loop pieces: the ant *sees where it is* from LiDAR + the grid,
and *ends the run when it enters the goal cell* (its scan matches the trained goal scan).

Both run ONLINE during a MuJoCo rollout, pure numpy, no torch. They use the same
``(x, y, theta)`` reference grid + histogram filter as the offline localization work
(``lidar_grid.py`` / ``lidar_localization.py``). Position is decoded from LiDAR only;
``(x, y)`` is used solely to pick the goal cell (exactly as the goal scan is already
synthesized from the goal position).

Two objects:
  * ``LiveLocalizer``  -- one predict+update histogram-filter step per scan, exposing
    the current MAP cell/belief ("where the ant thinks it is").
  * ``LidarArrivalDetector`` -- wraps a LiveLocalizer and declares ARRIVAL when the MAP
    cell is at the goal cell (per-theta match handled by the grid) AND, optionally, the
    current scan matches the trained goal scan within a tolerance. This is the PM's
    "entered the goal cell / scan matches the goal scan -> end the run" rule.
"""

import numpy as np

from .lidar_localization import GridLocalizer, scan_distance, _gauss_kernel1d


class LiveLocalizer:
    """Online histogram filter over the (x, y, theta) grid.

    Call :meth:`update` once per observed scan. Maintains a belief over all free poses
    (Markov/Bayes filter: predict = spatial+theta diffusion, update = measurement
    likelihood), and returns the MAP cell — the ant's LiDAR-decoded position.
    """

    def __init__(self, loc: GridLocalizer, motion_sigma_cells=1.0, theta_blur=1):
        self.loc = loc
        self.theta_blur = int(theta_blur)
        self._kernel = _gauss_kernel1d(float(motion_sigma_cells))
        self.reset()

    def reset(self, init_belief=None):
        if init_belief is None:
            b = self.loc._dense_free.astype(np.float64).copy()
        else:
            b = self.loc._to_dense(np.asarray(init_belief, dtype=np.float64))
        self.belief = b / max(b.sum(), 1e-30)
        self.n = 0
        self.est_cell = None
        self.est_xy = None
        self.est_theta = None

    def update(self, scan):
        """One filter step. Returns dict(est_cell, est_xy, est_theta, entropy, conf)."""
        scan = np.asarray(scan, dtype=np.float32)
        if self.n > 0:
            self.belief = self.loc._predict(self.belief, self._kernel, self.theta_blur)
        like = self.loc._to_dense(self.loc.likelihood(scan))
        b = self.belief * like
        s = b.sum()
        self.belief = (b / s) if s > 1e-30 else like / max(like.sum(), 1e-30)
        idx = np.unravel_index(int(np.argmax(self.belief)), self.belief.shape)
        x = float(self.loc.grid.x_vals[idx[1]]); y = float(self.loc.grid.y_vals[idx[0]])
        self.est_xy = (x, y)
        self.est_theta = float(self.loc.grid.theta_vals[idx[2]])
        self.est_cell = self.loc.grid.cell_of_xy(x, y)
        self.n += 1
        p = self.belief[self.belief > 0]
        entropy = float(-(p * np.log(p)).sum())
        conf = float(self.belief.max())                  # posterior mass at the MAP pose
        return dict(est_cell=self.est_cell, est_xy=self.est_xy, est_theta=self.est_theta,
                    entropy=entropy, conf=conf)


class LidarArrivalDetector:
    """Declare goal arrival from LiDAR alone (PM's end-of-run rule).

    Arrival is asserted when, for ``persist`` consecutive steps:
      1. the live MAP cell is within ``cell_radius`` of the goal cell (the ant has
         *entered the goal cell*; per-theta orientation match is handled by the grid's
         theta copies), AND
      2. (optional) the current scan matches the trained goal scan within ``scan_tol``
         per-beam RMS world-units (``None`` disables this extra gate; the cell match is
         the primary, orientation-robust criterion).

    ``goal_cell`` should be the grid cell of the goal — from localizing the goal scan
    (pure LiDAR) or the true goal (x, y). Keep the env's ground-truth success separately
    for evaluation so LiDAR-declared arrivals can be scored for false positives.
    """

    def __init__(self, loc: GridLocalizer, goal_scan, goal_cell, *, motion_sigma_cells=1.0,
                 theta_blur=1, cell_radius=0, scan_tol=None, persist=3, min_steps=5):
        self.live = LiveLocalizer(loc, motion_sigma_cells, theta_blur)
        self.goal_scan = np.asarray(goal_scan, dtype=np.float32)
        self.goal_cell = (int(goal_cell[0]), int(goal_cell[1]))
        self.cell_radius = int(cell_radius)
        self.scan_tol = scan_tol
        self.persist = int(persist)
        self.min_steps = int(min_steps)
        self._hit_run = 0
        self.arrived_step = None

    def reset(self):
        self.live.reset(); self._hit_run = 0; self.arrived_step = None

    def update(self, scan):
        """One step. Returns dict with est_cell, goal_scan_dist, and `arrived` bool."""
        st = self.live.update(scan)
        ec = st['est_cell']
        cell_ok = (abs(ec[0] - self.goal_cell[0]) <= self.cell_radius and
                   abs(ec[1] - self.goal_cell[1]) <= self.cell_radius)
        gd = float(scan_distance(scan, self.goal_scan[None])[0, 0])
        scan_ok = (self.scan_tol is None) or (gd <= float(self.scan_tol))
        hit = bool(cell_ok and scan_ok and self.live.n >= self.min_steps)
        self._hit_run = self._hit_run + 1 if hit else 0
        arrived = self._hit_run >= self.persist
        if arrived and self.arrived_step is None:
            self.arrived_step = self.live.n
        st.update(cell_ok=cell_ok, scan_ok=scan_ok, goal_scan_dist=gd, arrived=bool(arrived))
        return st
