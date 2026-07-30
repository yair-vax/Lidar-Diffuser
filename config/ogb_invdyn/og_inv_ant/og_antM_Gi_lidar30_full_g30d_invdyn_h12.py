"""
PROPRIOCEPTION LiDAR inverse-dynamics config -- the real fix.

The baseline LiDAR inv-dyn used obs_layout='xyo_lidar' = [x, y, cos, sin, lidar] (34-D),
which DROPS obs[7:] = the ant's 8 joint angles and all velocities. The horizon sweep
(lidar_invdyn_horizon_sweep.py) showed action R^2 plateaus at ~0.31 for EVERY goal
horizon -- an input-information ceiling, not a goal-posing problem: you cannot predict
8 leg torques from a car-like (x, y, theta) + lidar state that can't see the legs. Every
*working* inv-dyn in this repo uses the full 29-D ant observation.

This config restores that proprioception while keeping the LiDAR goal:

    obs   = obs_layout='full_lidar' = [full 29-D ant obs, lidar(30)]   -> obs_dim 59
    goal  = next LiDAR scan = the final 30 dims = goal_sel_idxs [29..58]
    target= the 8-D action (unchanged)
    model input_dim = obs_dim + len(goal_sel_idxs) = 59 + 30 = 89 (auto in train_og_invdyn)

Goal horizon: trainer_dict fixes it to k in {3, 4} (the sweep's sweet spot; k=1 was the
*worst* because a one-step goal scan barely moves). The planner is UNCHANGED -- it still
plans 30-beam LiDAR scans; only the inv-dyn that turns plans into torques changes. At
rollout the live env already provides the full 29-D obs, so the rollout subclass just
stops discarding it (handled by the 'full_lidar' branch in _obs_to_invdyn_obs).

config_fn -> logs to its own '...lidar30_full_g30d_invdyn_h12' dir; baseline untouched.
"""

import os.path as osp
import numpy as np

from diffuser.utils import watch

config_fn = osp.splitext(osp.basename(__file__))[0]

diffusion_args_to_watch = [
    ('prefix', ''),
    ('config_fn', config_fn),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
]

plan_args_to_watch = [
    ('prefix', ''),
    ('config_fn', config_fn),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ('value_horizon', 'V'),
    ('discount', 'd'),
    ('normalizer', ''),
    ('batch_size', 'b'),
    ('conditional', 'cond'),
]


tot_horizon = 12

## ---- LiDAR + proprioception observation settings ----
N_BEAMS = 30
RAW_OBS_DIM = 29                              # full OGBench ant obs (incl. joints + velocities)
LIDAR_CONFIG = dict(
    n_beams=N_BEAMS,
    max_range=12.0,
    fov=2.0 * np.pi,
    step=0.1,
    obs_layout='full_lidar',                  # [full raw obs, lidar...]  (the fix)
    raw_obs_dim=RAW_OBS_DIM,                  # so lidar slice = [29 .. 29+N_BEAMS)
    use_cache=True,                          # reuses the SAME cached scans as the baseline
)

obs_dim = RAW_OBS_DIM + N_BEAMS                          # composite obs dim (59)
goal_sel_idxs = tuple(range(RAW_OBS_DIM, RAW_OBS_DIM + N_BEAMS))  # goal = next LiDAR scan (30)


base = {
    'dataset': "antmaze-giant-stitch-v0",

    'diffusion': {
        'config_fn': '',

        'tot_horizon': tot_horizon,
        'goal_sel_idxs': goal_sel_idxs,

        ## inv model (unchanged architecture; input_dim grows to 89 automatically)
        'model': 'ogb_task.og_inv_dyn.MLP_InvDyn_OgB_V3',
        'inv_m_config': dict(
            hidden_dims=[512, 512, 512],
            final_fc_init_scale=1e-2,
            is_out_dist=False,
        ),
        'act_net_config': dict(
            act_f='gelu',
            use_dpout=False,
        ),

        ## well-posed goal horizon (the sweep's sweet spot k=3..4)
        'trainer_dict': dict(
            goal_horizon_mode='fixed',
            fixed_goal_k=3,
            goal_k_max=4,
        ),

        'renderer': 'guides.Maze2dRenderer_V2',

        ## dataset (LiDAR inverse-dynamics loader, now full_lidar layout)
        'loader': 'datasets.ogb_dset.OgB_Lidar_InvDyn_SeqDataset_V1',
        'termination_penalty': None,
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': [],
        'clip_denoised': True,
        'use_padding': True,
        'max_path_length': 220,
        'max_n_episodes': 10000,
        'dataset_config': dict(
            obs_select_dim=tuple(range(obs_dim)),  # ignored; lidar_config drives layout
            dset_type='ogb',
            lidar_config=LIDAR_CONFIG,
        ),

        ## serialization
        'logbase': 'logs',
        'prefix': 'diffusion/',
        'exp_name': watch(diffusion_args_to_watch),

        ## training (unchanged)
        'n_steps_per_epoch': 10000,
        'loss_type': 'l2_inv_v3',
        'n_train_steps': 2e6,

        'ema_decay': 0.995,
        'batch_size': 1024,
        'learning_rate': 1e-4,
        'gradient_accumulate_every': 1,

        'sample_freq': 4000,
        'save_freq': 4000,
        'n_saves': 5,

        'n_reference': 20,
        'n_samples': 8,

        'device': 'cuda',
    },

    'plan': {
        'config_fn': '',

        'batch_size': 1,
        'device': 'cuda',

        'horizon': tot_horizon,
        'n_diffusion_steps': 512,
        'normalizer': 'LimitsNormalizer',

        'vis_freq': 10,
        'logbase': 'logs',
        'prefix': 'plans/release',
        'exp_name': watch(plan_args_to_watch),
        'suffix': '0',

        'conditional': False,

        'diffusion_loadpath': 'f:diffusion/H{horizon}_T{n_diffusion_steps}',
        'diffusion_epoch': 'latest',
    },

}
