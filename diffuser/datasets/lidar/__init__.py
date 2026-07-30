"""
LiDAR-based observation toolkit for CompDiffuser / OGBench maze environments.

This package replaces the direct (x, y) position observation of the agent with a
simulated 2D LiDAR scan: a vector of ray distances measured (in the body frame of
the agent) against the maze walls. See ``lidar_sim.py`` for the simulator and
``LIDAR_README.md`` (repo root) for the overall design.
"""

from diffuser.datasets.lidar.lidar_sim import (
    MAZE_MAPS,
    LidarScanner,
    get_maze_map,
    quat_to_yaw,
    obs_to_xy_yaw,
)

__all__ = [
    'MAZE_MAPS',
    'LidarScanner',
    'get_maze_map',
    'quat_to_yaw',
    'obs_to_xy_yaw',
]
