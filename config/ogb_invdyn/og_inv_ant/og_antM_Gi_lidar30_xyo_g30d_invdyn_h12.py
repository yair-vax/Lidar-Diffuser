"""
Inverse-dynamics model for the LiDAR giant antmaze planner.

Spec (stage 5): "the function receives position (x, y), orientation (a car-like
model) and LiDAR readings, and an inverse-dynamics model extracts the action that
takes the agent from the current state to the next."

Accordingly the observation here is the composite
    [x, y, cos(yaw), sin(yaw), lidar_0 ... lidar_{N_BEAMS-1}]   (obs_dim = N_BEAMS + 4)
and the "goal" passed to the model is the *next* LiDAR scan (the quantity the
diffusion planner produces). We therefore set ``goal_sel_idxs`` to the LiDAR slice
of the observation, i.e. indices [4, 4 + N_BEAMS).

This mirrors ``config/ogb_invdyn/og_inv_ant/og_antM_Gi_o29d_g2d_invdyn_h12.py``;
only the observation/goal layout changed. The resulting checkpoints are what the
giant planner uses to turn generated LiDAR plans into executable actions.
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

## ---- LiDAR observation settings (must match the planner config) ----
N_BEAMS = 30
LIDAR_CONFIG = dict(
    n_beams=N_BEAMS,
    max_range=12.0,
    fov=2.0 * np.pi,
    step=0.1,
    obs_layout='xyo_lidar',   # [x, y, cos(yaw), sin(yaw), lidar...]
    use_cache=True,
)

obs_dim = N_BEAMS + 4                       # composite observation dim (34)
goal_sel_idxs = tuple(range(4, 4 + N_BEAMS))  # goal = next LiDAR scan (30 dims)


base = {
    'dataset': "antmaze-giant-stitch-v0",

    'diffusion': {
        'config_fn': '',

        'tot_horizon': tot_horizon,
        'goal_sel_idxs': goal_sel_idxs,

        ## inv model
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

        'trainer_dict': dict(),

        'renderer': 'guides.Maze2dRenderer_V2',

        ## dataset (LiDAR inverse-dynamics loader)
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
            ## --- LiDAR transform ---
            lidar_config=LIDAR_CONFIG,
        ),

        ## serialization
        'logbase': 'logs',
        'prefix': 'diffusion/',
        'exp_name': watch(diffusion_args_to_watch),

        ## training
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

        ## diffusion model
        'horizon': tot_horizon,
        'n_diffusion_steps': 512,
        'normalizer': 'LimitsNormalizer',

        ## serialization
        'vis_freq': 10,
        'logbase': 'logs',
        'prefix': 'plans/release',
        'exp_name': watch(plan_args_to_watch),
        'suffix': '0',

        'conditional': False,

        ## loading
        'diffusion_loadpath': 'f:diffusion/H{horizon}_T{n_diffusion_steps}',
        'diffusion_epoch': 'latest',
    },

}
