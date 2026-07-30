"""
2D LiDAR simulator for the OGBench maze environments (point / ant / humanoid maze).

Geometry follows the OGBench maze convention used throughout this repo
(see ``diffuser/utils/ogb_paper_vis_utils/ogb_vis_multi_agent_env.py`` and
``diffuser/datasets/ogb_dset/ogb_utils.py``):

    * ``maze_map[i, j] == 1`` is a solid wall cell, ``0`` is free space.
    * Each cell spans ``maze_unit`` world units and is axis aligned.
    * Cell ``(i, j)`` is centered at world coordinate
          ``x = j * maze_unit - offset_x`` ,  ``y = i * maze_unit - offset_y`` .
      Hence ``i`` indexes the world ``y`` axis and ``j`` indexes the world ``x`` axis.
    * The world point ``(x, y)`` lies in cell
          ``j = floor((x + offset_x + 0.5 * maze_unit) / maze_unit)``
          ``i = floor((y + offset_y + 0.5 * maze_unit) / maze_unit)`` .

The agent has a position ``(x, y)`` and a heading ``yaw`` (a "car-like" orientation).
We cast ``n_beams`` rays spread over ``fov`` radians *relative to the heading* and
return, for each ray, the distance to the first wall (clipped to ``max_range``).
This produces a body-frame, orientation-dependent observation vector, which is the
quantity used to replace the (x, y) coordinate in training.

The module is pure-numpy (no torch / mujoco / ogbench dependency) so it can be unit
tested and used for offline dataset generation on any machine.
"""

import numpy as np

# ---------------------------------------------------------------------------#
# Maze layouts. Copied verbatim from the OGBench maze definitions used in this
# repo (diffuser/utils/ogb_paper_vis_utils/ogb_vis_multi_agent_env.py). These are
# the single source of truth for the wall geometry, so the simulated LiDAR is
# exactly consistent with the dataset the agent was rolled out in.
# ---------------------------------------------------------------------------#

MAZE_MAP_ARENA = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]

MAZE_MAP_MEDIUM = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]

MAZE_MAP_LARGE = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

MAZE_MAP_GIANT = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1],
    [1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

# Keyed by the substring that appears in OGBench env names (antmaze-giant-... etc.)
MAZE_MAPS = {
    'arena': np.array(MAZE_MAP_ARENA, dtype=np.int8),
    'medium': np.array(MAZE_MAP_MEDIUM, dtype=np.int8),
    'large': np.array(MAZE_MAP_LARGE, dtype=np.int8),
    'giant': np.array(MAZE_MAP_GIANT, dtype=np.int8),
}


def get_maze_map(env_name):
    """Return the (H, W) occupancy grid for an OGBench env name like
    ``antmaze-giant-stitch-v0``.  ``1`` == wall, ``0`` == free."""
    name = str(env_name).lower()
    for key, mp in MAZE_MAPS.items():
        if f'-{key}' in name or f'_{key}' in name or key in name:
            return mp
    raise ValueError(
        f'Could not infer maze layout from env name "{env_name}". '
        f'Known maze types: {list(MAZE_MAPS.keys())}')


# ---------------------------------------------------------------------------#
# Orientation helpers
# ---------------------------------------------------------------------------#

def quat_to_yaw(quat):
    """Convert a (..., 4) quaternion in (w, x, y, z) order (MuJoCo convention,
    matching OGBench ``obs[..., 3:7]``) into a yaw angle (rotation about world z).

    yaw = atan2( 2 (w z + x y), 1 - 2 (y^2 + z^2) )
    """
    quat = np.asarray(quat, dtype=np.float64)
    w = quat[..., 0]
    x = quat[..., 1]
    y = quat[..., 2]
    z = quat[..., 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)


def obs_to_xy_yaw(observations, has_quat=True):
    """Extract ``(xy, yaw)`` from a batch of OGBench ant/humanoid observations.

    The OGBench ant observation layout is::
        obs[0:2]  -> x, y          (global position)
        obs[2]    -> z height
        obs[3:7]  -> torso quaternion (w, x, y, z)
        obs[7:]   -> joint angles + velocities

    Args:
        observations: (N, D) array. Requires D >= 7 when ``has_quat``.
        has_quat: if False (e.g. pointmaze, D == 2), yaw is taken to be 0.
    Returns:
        xy:  (N, 2) float64
        yaw: (N,)   float64
    """
    obs = np.asarray(observations, dtype=np.float64)
    xy = obs[..., 0:2]
    if has_quat and obs.shape[-1] >= 7:
        yaw = quat_to_yaw(obs[..., 3:7])
    else:
        yaw = np.zeros(obs.shape[:-1], dtype=np.float64)
    return xy, yaw


# ---------------------------------------------------------------------------#
# LiDAR scanner
# ---------------------------------------------------------------------------#

class LidarScanner:
    """Ray-cast a 2D LiDAR against a maze occupancy grid.

    Args:
        maze_map: (H, W) int array, 1 == wall.
        n_beams: number of rays (== output dimension of the LiDAR observation).
        max_range: maximum sensing distance in world units. Beams that hit
            nothing return ``max_range``.
        fov: angular field of view in radians. ``2*pi`` == full 360 degree scan.
        fov_centered: if True the beams are spread symmetrically around the
            heading over ``[-fov/2, +fov/2]`` (natural for a forward facing
            sensor with ``fov < 2*pi``). Ignored / irrelevant for a 360 scan.
        maze_unit, offset_x, offset_y: OGBench world<->cell geometry constants.
        step: ray-march resolution in world units. Smaller == more accurate /
            slower. ``0.1`` (== 2.5% of a cell) is plenty.
    """

    def __init__(self, maze_map, n_beams=30, max_range=12.0, fov=2.0 * np.pi,
                 fov_centered=False, maze_unit=4.0, offset_x=4.0, offset_y=4.0,
                 step=0.1):
        self.maze_map = np.asarray(maze_map, dtype=np.int8)
        assert self.maze_map.ndim == 2
        self.H, self.W = self.maze_map.shape
        self.n_beams = int(n_beams)
        self.max_range = float(max_range)
        self.fov = float(fov)
        self.fov_centered = bool(fov_centered)
        self.maze_unit = float(maze_unit)
        self.offset_x = float(offset_x)
        self.offset_y = float(offset_y)
        self.step = float(step)

        # Per-beam angular offsets relative to the agent heading.
        is_full_circle = np.isclose(self.fov, 2.0 * np.pi)
        if is_full_circle:
            # endpoint=False so we do not duplicate the 0 / 2pi beam.
            self.beam_offsets = np.linspace(
                0.0, 2.0 * np.pi, self.n_beams, endpoint=False)
        elif self.fov_centered:
            self.beam_offsets = np.linspace(
                -self.fov / 2.0, self.fov / 2.0, self.n_beams)
        else:
            self.beam_offsets = np.linspace(0.0, self.fov, self.n_beams)
        self.beam_offsets = self.beam_offsets.astype(np.float64)

    # -- geometry ----------------------------------------------------------#
    def world_to_cell(self, x, y):
        """Vectorized world -> integer cell indices (i over y, j over x)."""
        j = np.floor((x + self.offset_x + 0.5 * self.maze_unit) / self.maze_unit)
        i = np.floor((y + self.offset_y + 0.5 * self.maze_unit) / self.maze_unit)
        return i.astype(np.int64), j.astype(np.int64)

    def _is_wall(self, i, j):
        """Wall test that treats out-of-bounds as a wall (the maze is fully
        enclosed by border walls, so this only matters numerically)."""
        oob = (i < 0) | (i >= self.H) | (j < 0) | (j >= self.W)
        ic = np.clip(i, 0, self.H - 1)
        jc = np.clip(j, 0, self.W - 1)
        wall = self.maze_map[ic, jc] == 1
        return wall | oob

    # -- scanning ----------------------------------------------------------#
    def scan_batch(self, xy, yaw):
        """Compute LiDAR scans for a batch of poses.

        Args:
            xy:  (N, 2) world positions.
            yaw: (N,)   headings in radians.
        Returns:
            (N, n_beams) float32 distances in world units, in [0, max_range].
        """
        xy = np.asarray(xy, dtype=np.float64)
        yaw = np.asarray(yaw, dtype=np.float64)
        assert xy.ndim == 2 and xy.shape[1] == 2
        N = xy.shape[0]
        if N == 0:
            return np.zeros((0, self.n_beams), dtype=np.float32)

        # (N, n_beams) ray directions.
        angles = yaw[:, None] + self.beam_offsets[None, :]
        dir_x = np.cos(angles)
        dir_y = np.sin(angles)

        x0 = xy[:, 0:1]
        y0 = xy[:, 1:2]

        dist = np.full((N, self.n_beams), self.max_range, dtype=np.float64)
        active = np.ones((N, self.n_beams), dtype=bool)

        # March outward in fixed steps. We start one step out (the origin is in
        # a free cell by construction) and stop at max_range.
        n_steps = int(np.ceil(self.max_range / self.step))
        for s_idx in range(1, n_steps + 1):
            t = s_idx * self.step
            if t > self.max_range:
                t = self.max_range
            px = x0 + t * dir_x
            py = y0 + t * dir_y
            i, j = self.world_to_cell(px, py)
            wall = self._is_wall(i, j)
            newly_hit = active & wall
            if newly_hit.any():
                dist[newly_hit] = t
                active[newly_hit] = False
            if not active.any():
                break

        return dist.astype(np.float32)

    def scan(self, x, y, yaw):
        """Convenience single-pose scan -> (n_beams,) float32."""
        out = self.scan_batch(np.array([[x, y]], dtype=np.float64),
                              np.array([yaw], dtype=np.float64))
        return out[0]

    def scan_obs(self, observations, has_quat=True, chunk_size=100_000):
        """Compute LiDAR scans directly from raw OGBench observations.

        Processes in chunks to bound peak memory on the full (1M+) dataset.

        Args:
            observations: (N, D) OGBench observations.
            has_quat: whether obs carries a quaternion at obs[3:7] (ant/humanoid).
            chunk_size: number of rows processed per vectorized call.
        Returns:
            (N, n_beams) float32 LiDAR scans.
        """
        observations = np.asarray(observations)
        N = observations.shape[0]
        out = np.empty((N, self.n_beams), dtype=np.float32)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            xy, yaw = obs_to_xy_yaw(observations[start:end], has_quat=has_quat)
            out[start:end] = self.scan_batch(xy, yaw)
        return out

    # -- metadata ----------------------------------------------------------#
    def config_dict(self):
        return dict(
            n_beams=self.n_beams,
            max_range=self.max_range,
            fov=self.fov,
            fov_centered=self.fov_centered,
            maze_unit=self.maze_unit,
            offset_x=self.offset_x,
            offset_y=self.offset_y,
            step=self.step,
            maze_shape=list(self.maze_map.shape),
        )


def make_scanner_for_env(env_name, lidar_config=None):
    """Factory that builds a :class:`LidarScanner` from an OGBench env name and
    an optional config dict (the ``lidar_config`` key used in dataset configs)."""
    lidar_config = dict(lidar_config or {})
    maze_map = get_maze_map(env_name)
    return LidarScanner(
        maze_map=maze_map,
        n_beams=lidar_config.get('n_beams', 30),
        max_range=lidar_config.get('max_range', 12.0),
        fov=lidar_config.get('fov', 2.0 * np.pi),
        fov_centered=lidar_config.get('fov_centered', False),
        maze_unit=lidar_config.get('maze_unit', 4.0),
        offset_x=lidar_config.get('offset_x', 4.0),
        offset_y=lidar_config.get('offset_y', 4.0),
        step=lidar_config.get('step', 0.1),
    )
