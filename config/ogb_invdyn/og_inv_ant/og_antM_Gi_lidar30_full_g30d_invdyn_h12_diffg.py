"""
v2a -- DIFFERENCE-GOAL LiDAR inverse dynamics (2026-07-07): same recipe as the
proven h12 run (k in {3,4}, full_lidar 59-D obs), but the model's goal input is
    g_input = normed(goal scan) - normed(current scan)
instead of the absolute goal scan.

Why (see LIDAR_DIAGNOSIS_AND_BASELINE.md 2026-07-07 entries): the h12 model
reads its absolute-scan goal at only ~20% of action variance (stop 0.11, turn
0.16/garbage) and NO eval-time input fixed it (A2 real-scan ribbons, action-CFG
w3/w6, bands, yaw -- ~250 episodes, all flat at ~48 mj). The wide-k retrain
made it fully goal-blind (shuffled delta 0.070 at k=4 AND k=30). Hypothesis
v2a tests: the barrier is DECODABILITY -- the original xy-goal inv-dyn steered
this executor because direction was trivially readable (delta-xy); an absolute
scan pair requires implicit relative-pose estimation the MLP never learned.
The difference scan restores that readability: "stay" becomes the ZERO vector
(intrinsic stop signal), translation/rotation become structured sign patterns.
k stays {3,4} exactly -- widening k is REFUTED (it dilutes the informative
near-k mass; the widek run is the proof).

ACCEPTANCE (probe auto-detects the encoding from this run's trainer pkl):
    python -m diffuser.pl_eval.lidar_eval.lidar_invdyn_goal_sensitivity \
        --inv_model_path logs/antmaze-giant-stitch-v0/diffusion/og_antM_Gi_lidar30_full_g30d_invdyn_h12_diffg \
        --inv_epoch latest --device cpu --out logs/lidar_eval/invdyn_goal_sensitivity_diffg.json
    PASS = shuffled delta >= ~0.4, self >= ~0.2, roll2/roll7 nonzero & ordered,
           true-MSE@k4 <= ~0.10.  (h12 baseline: 0.250 / 0.113 / 0.157-0.286 / 0.045)
Then closed-loop: LIDAR_INV_MODEL_PATH=<this run dir> sbatch plan_lidar_a2.sbatch
(rollout auto-detects goal_encoding='diff' from the trainer pkl; A2 real-scan
ribbon + LOOKAHEAD=3 is the matched executor).

Everything else (obs, dims, arch, lr, steps, normalizer) is identical to
og_antM_Gi_lidar30_full_g30d_invdyn_h12. New config_fn -> logs to its own
'..._invdyn_h12_diffg' dir; the h12 run is untouched.
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


tot_horizon = 12                              # same window as the proven h12 run

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

        ## v2a: k = {3,4} exactly like h12; goal input = diff scan (the change)
        'trainer_dict': dict(
            goal_horizon_mode='fixed',
            fixed_goal_k=3,
            goal_k_max=4,
            goal_encoding='diff',
        ),

        'renderer': 'guides.Maze2dRenderer_V2',

        ## dataset (LiDAR inverse-dynamics loader, full_lidar layout; == h12)
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
