"""
GOAL-AWARE (wide-k) LiDAR inverse-dynamics config -- makes the goal channel earn
its gradient. (2026-07-05)

Why: the goal-sensitivity probe (invdyn_goal_sensitivity.json, k=4, 102k trans)
showed the h12 model reads its goal but only weakly: the goal channel explains
~20% of action variance (true 0.045 -> shuffled 0.134, var 0.453) while
proprioception alone explains 70%; the stop-vs-go contrast is 0.11 of action RMS,
a 24-deg turn command 0.16, and an 84-deg turn is treated like garbage (0.141 ~=
zeros 0.123) -- i.e., off-path goals are gracefully IGNORED. Closed loop the
executor's subgoals live exactly in that whisper/ignored regime, which is why all
six gridnav arms were statistically identical (plan content irrelevant).

Root cause of the weakness: with k in {3, 4} the future scan is ~always a small
along-path shift of the current one -- turns and diverging futures never appear,
so the goal never disambiguates direction at training time. The stitch data's
noisy waypoint expert DOES leave a direction-multimodal a|s (that residual is the
20%), but only wide horizons reveal it.

Fix here (config-only; trainer already supports it): goal k ~ Uniform[1, 60]
(dataset window h64). At k up to 60 (~2 cells of travel) real futures include
corner turns and diverging routes, so the goal scan carries direction the state
lacks -- turn/far goals become in-distribution WITH ground-truth actions.

ACCEPTANCE (run lidar_invdyn_goal_sensitivity.py on this run when trained):
    shuffled act_delta_ratio  >= ~0.4   (was 0.25)
    roll2 / roll7 deltas      clearly nonzero & ordered (was 0.16 / ignored)
    self-vs-true delta        >= ~0.2   (was 0.11)
    true-goal MSE at k=4      <= ~0.10  (was 0.045; some rise is the price)
Then closed-loop nav_true (pursuit v2, LIDAR_NAV_LOOKAHEAD can rise to ~30-50 to
exploit the wider horizon). If deltas still < 0.3 -> v2 = difference-goal
encoding (feed g - current_scan; trainer + rollout change) per the memory file.

Everything else (obs 59-D full_lidar, goal = trailing 30 scan dims, model, lr,
steps) is identical to og_antM_Gi_lidar30_full_g30d_invdyn_h12. New config_fn ->
logs to its own '..._invdyn_h64_widek' dir; the h12 run is untouched. Train with
train_lidar_invdyn.sbatch --config pointed here.
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


tot_horizon = 64                              # dataset window; goal k drawn in [1, 60]

## ---- LiDAR + proprioception observation settings (unchanged from h12) ----
N_BEAMS = 30
RAW_OBS_DIM = 29                              # full OGBench ant obs (incl. joints + velocities)
LIDAR_CONFIG = dict(
    n_beams=N_BEAMS,
    max_range=12.0,
    fov=2.0 * np.pi,
    step=0.1,
    obs_layout='full_lidar',                  # [full raw obs, lidar...]
    raw_obs_dim=RAW_OBS_DIM,                  # so lidar slice = [29 .. 29+N_BEAMS)
    use_cache=True,                           # reuses the SAME cached scans
)

obs_dim = RAW_OBS_DIM + N_BEAMS                                   # 59
goal_sel_idxs = tuple(range(RAW_OBS_DIM, RAW_OBS_DIM + N_BEAMS))  # goal = LiDAR scan (30)


base = {
    'dataset': "antmaze-giant-stitch-v0",

    'diffusion': {
        'config_fn': '',

        'tot_horizon': tot_horizon,
        'goal_sel_idxs': goal_sel_idxs,

        ## inv model (unchanged architecture; input_dim 89 auto in train_og_invdyn)
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

        ## GOAL-AWARE horizon: k ~ Uniform[1, 60] (the whole point of this config)
        'trainer_dict': dict(
            goal_horizon_mode='fixed',
            fixed_goal_k=1,
            goal_k_max=60,
        ),

        'renderer': 'guides.Maze2dRenderer_V2',

        ## dataset (LiDAR inverse-dynamics loader, full_lidar layout)
        'loader': 'datasets.ogb_dset.OgB_Lidar_InvDyn_SeqDataset_V1',
        'termination_penalty': None,
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': [],
        'clip_denoised': True,
        'use_padding': True,
        ## h64 needs >= path_len-1+horizon = 199+64 = 263 (make_indices asserts
        ## max_path_length - horizon >= path_length - 1 so every transition is
        ## included; 220 was fine for h12 but crashes h64). Padding is zero-fill
        ## and val_len-masked, and the normalizer only sees valid steps, so norm
        ## constants stay identical to the h12 run.
        'max_path_length': 264,
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
