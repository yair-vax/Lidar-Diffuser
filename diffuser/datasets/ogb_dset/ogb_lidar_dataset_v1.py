"""
LiDAR dataset entry points.

These are *thin* subclasses of the existing, well-tested OGBench dataset classes.
They add no new logic: the LiDAR transform happens inside
``ogb_sequence_dataset`` (see ogb_utils.py) whenever ``dataset_config`` carries a
``'lidar_config'`` entry. The subclasses exist only to give the LiDAR pipeline
clearly named, self-documenting loaders that the config files reference, and to
sanity-check that a ``lidar_config`` was actually provided.

  * ``OgB_Lidar_SeqDataset_V1``        -> diffusion planner (observation = LiDAR,
                                          no x/y given to the model).
  * ``OgB_Lidar_InvDyn_SeqDataset_V1`` -> inverse-dynamics model (observation =
                                          [x, y, cos(yaw), sin(yaw), LiDAR]).
"""

import diffuser.utils as utils
from diffuser.datasets.ogb_dset.ogb_dataset_v2 import OgB_SeqDataset_V2
from diffuser.datasets.ogb_dset.ogb_invdyn_dataset_v1 import OgB_InvDyn_SeqDataset_V1


def _check_lidar_config(dataset_config):
    lidar_config = dataset_config.get('lidar_config', None)
    assert lidar_config is not None, (
        'LiDAR dataset requires dataset_config["lidar_config"]; '
        'use the regular OgB_*SeqDataset class for raw (x, y) observations.')
    utils.print_color(f'[lidar dataset] lidar_config={lidar_config}', c='c')
    return lidar_config


class OgB_Lidar_SeqDataset_V1(OgB_SeqDataset_V2):
    """Diffusion-planner dataset whose observations are simulated LiDAR scans."""

    def __init__(self, *args, dataset_config={}, **kwargs):
        _check_lidar_config(dataset_config)
        super().__init__(*args, dataset_config=dataset_config, **kwargs)


class OgB_Lidar_InvDyn_SeqDataset_V1(OgB_InvDyn_SeqDataset_V1):
    """Inverse-dynamics dataset whose observations carry the LiDAR scan.

    Two layouts are supported:
      * ``'xyo_lidar'``  : [x, y, cos(yaw), sin(yaw), LiDAR]  (the spec's car-like
                           "position, orientation and LiDAR" -- but it omits the
                           ant's joints/velocities, which caps inverse-dynamics
                           accuracy; see LIDAR_DIAGNOSIS_AND_BASELINE.md).
      * ``'full_lidar'`` : [full raw obs (incl. joints + velocities), LiDAR]  --
                           restores proprioception so the model can actually
                           predict the 8 leg torques.
    The goal (next LiDAR scan) is the final n_beams dims in both cases."""

    def __init__(self, *args, dataset_config={}, **kwargs):
        lidar_config = _check_lidar_config(dataset_config)
        assert lidar_config.get('obs_layout') in ('xyo_lidar', 'full_lidar'), (
            'inverse-dynamics LiDAR dataset expects obs_layout="xyo_lidar" or '
            '"full_lidar" so the model receives the agent state + LiDAR.')
        super().__init__(*args, dataset_config=dataset_config, **kwargs)
