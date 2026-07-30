"""
v2b -- ACTION-CHUNK LiDAR inverse dynamics (2026-07-07): predict the next
m=16 actions (flattened, val_len-masked L2) from (state, diff-encoded goal
scan at k ~ U[12, 20]), instead of only the first action.

Why (the structural fix; see LIDAR_DIAGNOSIS_AND_BASELINE.md 2026-07-07):
a single-action inverse-dynamics target gives the goal NO necessary role at
any horizon -- at k<=4 the proprio's root velocities (momentum) already
predict the k-step future, so the goal is redundant; at k>=~15 the FIRST
torque is goal-independent, so the goal is uninformative. h12's 20% whisper
and widek's total blindness are both correct regression answers to that
target. The chunk changes the target itself: the next 16 actions (~2 wu of
travel) CONTAIN the turn toward the goal, so I(goal; a_{1:16} | state) is
large -- the loss finally FORCES the goal channel to earn its gradient.
The goal is diff-encoded (g - current scan) so direction is also trivially
decodable (v2a's lever, stacked for maximum win probability).

Execution: receding horizon -- the rollout auto-detects act_chunk from this
run's trainer pkl, queries the model, plays the first LIDAR_ACT_CHUNK_EXEC
(default m/2 = 8) actions, then re-queries. Chunk commitment also drowns the
per-step gait pull that has dominated every closed-loop run so far.

ACCEPTANCE (probe auto-detects encoding + chunk; use k near the trained mid):
    python -m diffuser.pl_eval.lidar_eval.lidar_invdyn_goal_sensitivity \
        --inv_model_path logs/antmaze-giant-stitch-v0/diffusion/og_antM_Gi_lidar30_full_g30d_invdyn_h24_chunk16 \
        --inv_epoch latest --device cpu --k 16 --out logs/lidar_eval/invdyn_goal_sensitivity_chunk16.json
    PASS = shuffled TAIL delta >= ~0.4 with per-index deltas RISING along the
    chunk (first < mid < tail); proprio_shuf still explodes.
Then closed-loop: LIDAR_INV_MODEL_PATH=<this run dir> LIDAR_NAV_LOOKAHEAD=16
sbatch plan_lidar_a2.sbatch (A2 real-scan ribbon; act-CFG is skipped for
chunk models by design).

Window h=24 (>= goal_k_max 20 + 1 and >= m 16). max_path_length = 199 + 24 =
223 -> 224 (the make_indices constraint that crashed h64 at 220). ACT_CHUNK
is defined ONCE below and threaded into both the model head sizing
(diffusion['act_chunk']) and the trainer target (trainer_dict['act_chunk']).
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


tot_horizon = 24                              # window >= max(goal k)+1 and >= chunk m
ACT_CHUNK = 16                                # m: actions predicted per query (~2 wu)

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

        ## model head widened to m*8 by train_og_invdyn via act_chunk
        'model': 'ogb_task.og_inv_dyn.MLP_InvDyn_OgB_V3',
        'act_chunk': ACT_CHUNK,
        'inv_m_config': dict(
            hidden_dims=[512, 512, 512],
            final_fc_init_scale=1e-2,
            is_out_dist=False,
        ),
        'act_net_config': dict(
            act_f='gelu',
            use_dpout=False,
        ),

        ## v2b: goal at k ~ U[12, 20] (aligned with the chunk span), diff-encoded,
        ## target = next ACT_CHUNK actions
        'trainer_dict': dict(
            goal_horizon_mode='fixed',
            fixed_goal_k=12,
            goal_k_max=20,
            goal_encoding='diff',
            act_chunk=ACT_CHUNK,
        ),

        'renderer': 'guides.Maze2dRenderer_V2',

        ## dataset (LiDAR inverse-dynamics loader, full_lidar layout)
        'loader': 'datasets.ogb_dset.OgB_Lidar_InvDyn_SeqDataset_V1',
        'termination_penalty': None,
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': [],
        'clip_denoised': True,
        'use_padding': True,
        ## make_indices needs max_path_length - horizon >= path_length - 1 = 199
        ## (the h64 lesson); padding is zero-filled, val_len-masked, and the
        ## normalizer only sees valid steps -> norm constants match the h12 run.
        'max_path_length': 224,
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
