import sys, os; sys.path.append('./')
os.environ['PYOPENGL_PLATFORM'] = 'egl' ## enable GPU rendering in mujoco
os.environ['MUJOCO_GL'] = 'egl'
import pdb, torch, copy, pdb, json
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.use_deterministic_algorithms(True)
##
import numpy as np
np.set_printoptions(precision=3, suppress=True)
from datetime import datetime
import os.path as osp
import diffuser.utils as utils
from diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_planner_v1 import OgB_Stgl_Sml_MazeEnvPlanner_V1




class Parser(utils.Parser):
    dataset: str = None
    config: str = None
    ## should not put any existing variable in config file here
    pl_seeds: str = '-1' # no seed
    plan_n_ep: int = -100 ## all if -100, auto parse to int

def main(args_train, args):

    #---------------------------------- setup ----------------------------------#

    ld_config = dict()

    ## LiDAR planner: observations are 30-beam scans, so use the LiDAR rollout
    ## subclass (inserts scanner.scan for the planner conditioning + the 34-D
    ## inv-dyn obs). Selected by the presence of a lidar_config in the dataset.
    ## repeated-episode heatmap runs (2026-07-15): override which eval problem(s)
    ## to run, so the SAME start/goal can be repeated under many seeds without
    ## touching the per-config hardcoded ep_st_idx below. Unset (default) ->
    ## no-op, exact prior behavior.
    if 'LIDAR_EP_ST_IDX' in os.environ:
        args.ep_st_idx = int(os.environ['LIDAR_EP_ST_IDX'])

    is_lidar = args_train.dataset_config.get('lidar_config', None) is not None
    if is_lidar:
        from diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_lidar_planner_v1 import \
            OgB_Stgl_Sml_Lidar_MazeEnvPlanner_V1
        ogmz_planner = OgB_Stgl_Sml_Lidar_MazeEnvPlanner_V1(args_train, args=args)
    else:
        ogmz_planner = OgB_Stgl_Sml_MazeEnvPlanner_V1(args_train, args=args)
    ogmz_planner.setup_load( ld_config=ld_config )

    #---------------------------- start planning -----------------------------#

    pl_seeds = args.pl_seeds

    from diffuser.datasets.d4rl import Is_OgB_Robot_Env
    if len(pl_seeds) == 1:
        ## plan_n_ep
        if Is_OgB_Robot_Env:
            if pl_seeds[0] == -1: ## no seed
                avg_result_dict = ogmz_planner.ogb_plan_once(pl_seed=None,)
            else:
                avg_result_dict = ogmz_planner.ogb_plan_once(pl_seed=pl_seeds[0])

    else:
        ## multi-seed: repeat the SAME episode(s) (ep_st_idx..ep_st_idx+plan_n_ep)
        ## once per seed, reusing the already-loaded planner/model instead of
        ## reloading per seed. Stores the FULL per-episode ep_is_suc/
        ## ep_min_goal_dist for every seed -- general enough for both a single-
        ## episode repeat (plan_n_ep=1, e.g. a heatmap of one problem) and a
        ## full-benchmark multi-seed sweep (plan_n_ep=20, e.g. a fair baseline-
        ## vs-fix comparison -- a single seed per problem is too noisy to trust,
        ## measured directly: identical total (2/20) across two configs with a
        ## completely different SET of which episodes succeeded).
        utils.print_color(f'{args.pl_seeds=}')
        per_seed_summary = []
        base_savepath = ogmz_planner.savepath.rstrip(os.sep)
        for s in pl_seeds:
            ## self.savepath is set once at planner init and reused verbatim by
            ## ogb_plan_once() for every save + the final rename-to "-sr{rate}"
            ## step; without a per-call override, seed 2's rename collides with
            ## seed 1's already-renamed directory (OSError: Directory not empty).
            ogmz_planner.savepath = f'{base_savepath}_seed{s}'
            res = ogmz_planner.ogb_plan_once(pl_seed=(None if s == -1 else s))
            ep_is_suc = [bool(x) for x in res['ep_is_suc']]
            ep_min_goal_dist = [float(x) for x in res['ep_min_goal_dist']]
            per_seed_summary.append(dict(
                seed=int(s), ep_is_suc=ep_is_suc, ep_min_goal_dist=ep_min_goal_dist))
            utils.print_color(
                f'[multi-seed] seed={s} done ({len(per_seed_summary)}/{len(pl_seeds)}) '
                f'-> {sum(ep_is_suc)}/{len(ep_is_suc)} succeeded', c='c')
        avg_result_dict = dict(
            pl_seeds=list(pl_seeds),
            ep_st_idx=ogmz_planner.ep_st_idx,
            per_seed_summary=per_seed_summary,
        )
        heatmap_out = os.environ.get('LIDAR_HEATMAP_OUT', None)
        if heatmap_out:
            os.makedirs(os.path.dirname(heatmap_out) or '.', exist_ok=True)
            with open(heatmap_out, 'w') as f:
                json.dump(avg_result_dict, f, indent=2)
            utils.print_color(f'[multi-seed] saved -> {heatmap_out}', c='g')

    ## might prevent the final exception before the program finishes
    ogmz_planner.env.close()
    del ogmz_planner.env
    del ogmz_planner.renderer.env
    
    return avg_result_dict


if __name__ == '__main__':
    ## training args
    args_train = Parser().parse_args('diffusion')
    args = Parser().parse_args('plan')
    ## 1. get epoch to eval on, by default all
    loadpath = args.logbase, args.dataset, args_train.exp_name

    args.pl_seeds = utils.parse_seeds_str(args.pl_seeds) ## a list of int
    args.n_batch_acc_probs = 4 ##
    
    ### --- Hyper-parameters Setup ---
    from diffuser.datasets.d4rl import Is_OgB_Robot_Env
    assert Is_OgB_Robot_Env
    
    

    ## Default
    args.is_replan = None ## placeholder, should be replaced in the if code blcok below
    args.n_act_per_waypnt = 2
    args.is_save_pkl = False
    args.is_rd_agv = False

    ## the state dimension used in the diffusion models
    dfu_ndim = len(args_train.dataset_config['obs_select_dim'])

    ## LiDAR planner: the diffusion observation is a LiDAR scan, not (x, y), so we
    ## branch on the presence of a lidar_config (obs_select_dim is a placeholder
    ## (0, 1) in the LiDAR config and would otherwise collide with the 2-D branch).
    is_lidar_cfg = args_train.dataset_config.get('lidar_config', None) is not None


    ## ---------------------------------------
    ## ----------- Ant Maze Stitch -----------
    if 'antmaze' in args.dataset.lower() and 'stitch' in  args.dataset.lower():
        if 'giant' in args.dataset:

            ## Ant Maze Giant -- LiDAR closed-loop rollout (Option D)
            if is_lidar_cfg:
                ## First working video: one composed LiDAR plan per episode, the
                ## inv-dyn chases its scans. Adaptive replanning and the subgoal
                ## marker need per-waypoint (x, y), which a LiDAR plan does not
                ## carry, so they are disabled here (see LIDAR_ROLLOUT_VIDEO_TODO.md
                ## §3.2/§4/§7).
                args.ep_st_idx = 0
                repl_wp_cfg = {}
                args.n_act_per_waypnt = 1
                args.ev_cp_infer_t_type = 'interleave'
                args.ev_n_comp = 9
                args.rd_resol = 300
                args.is_use_subgoal_marker = False

                ## ---- tunable rollout knobs (env-driven so the baseline and the
                ## ---- variants run without re-editing; defaults == the first run) ----
                ## Goal-scan heading: '0.0' (fixed, default) or 'goal' (use the goal
                ## state's real heading -- sharper but pose-dependent conditioning).
                _yaw = os.environ.get('LIDAR_GOAL_YAW', '0.0')
                try:
                    args.lidar_goal_yaw = float(_yaw)
                except ValueError:
                    args.lidar_goal_yaw = _yaw                  # e.g. 'goal'

                ## Replanning: OFF by default (one plan per episode). Set
                ## LIDAR_REPLAN=1 to re-condition on the CURRENT scan whenever the
                ## current plan is consumed -- helps cross the giant maze, where a
                ## single H160 plan can't reach the goal. (mode 'periodic', added to
                ## ogb_stgl_sml_planner_v1.py; scan-space safe, keeps full n_comp.)
                ##
                ## LIDAR_REPLAN_MODE selects HOW replans beyond "plan consumed" get
                ## triggered:
                ##   'periodic' (default): also force a replan every LIDAR_REPL_EVERY
                ##     consumed waypoints regardless of progress -- discards the
                ##     current route and computes a brand new BFS route from scratch
                ##     even if nothing has gone wrong.
                ##   'dist_check' (2026-08-01): sticks to ONE route; only replans when
                ##     the ant's OWN scan-based position estimate drifts more than
                ##     LIDAR_ADA_DIST_THRES from where the route says it should be
                ##     (see _lidar_route_deviation) -- never a fixed clock, never
                ##     ground-truth (x,y).
                if os.environ.get('LIDAR_REPLAN', '0') == '1':
                    _repl_mode = os.environ.get('LIDAR_REPLAN_MODE', 'periodic').lower()
                    args.is_replan = 'lidar_dist_check' if _repl_mode == 'dist_check' else 'periodic'
                    args.repl_ada_dist_cfg = dict(
                        n_max_steps=int(os.environ.get('LIDAR_N_MAX_STEPS', '2500')),
                        max_n_repl=int(os.environ.get('LIDAR_MAX_N_REPL', '15')),
                        ## LIDAR_REPL_EVERY=N (>0): also replan every N consumed
                        ## waypoints, not only when the whole plan is used up.
                        ## 0 (default) keeps the original "replan when plan consumed".
                        ## Only read by 'periodic'; 'dist_check' ignores it entirely.
                        repl_every=int(os.environ.get('LIDAR_REPL_EVERY', '0')),
                        ## Only read by 'dist_check': deviation threshold, world units.
                        thres=float(os.environ.get('LIDAR_ADA_DIST_THRES', '4.0')),
                    )
                else:
                    args.is_replan = False
                    ## no-replan path: only n_max_steps is read from this dict
                    args.repl_ada_dist_cfg = dict(n_max_steps=2000)

                ## Multi-scan goal-band spacing (eval-only): override the config's
                ## goal_band_step so it can be SWEPT without editing the config or
                ## retraining. NOTE: goal_band_k (the trained band width) is NOT
                ## overridable here -- it must match the planner checkpoint, so it
                ## stays config-driven.
                args.goal_band_step = float(
                    os.environ.get('LIDAR_GOAL_BAND_STEP', getattr(args, 'goal_band_step', 4.0)))

                ## Teacher-forced REAL goal band (decisive diagnostic): LIDAR_TF_BAND=1
                ## replaces the SYNTHESIZED band with the TRUE last-K scans of a real
                ## dataset trajectory ending at the goal -> isolates "band concept" from
                ## "eval-time synthesis is OOD". LIDAR_TF_NPZ overrides the default
                ## ~/.ogbench/data/<env>-val.npz.
                args.tf_band = int(os.environ.get('LIDAR_TF_BAND', '0'))
                args.tf_npz = os.environ.get('LIDAR_TF_NPZ', '')
                ## Wide-baseline teacher band: frames between band scans (S>1 spaces
                ## the K real scans over SEVERAL cells of the approach instead of the
                ## last K consecutive frames). Eval-only; ignored unless LIDAR_TF_BAND=1.
                args.tf_band_stride = int(os.environ.get('LIDAR_TF_BAND_STRIDE', '1'))

                ## --- localization-guided plan re-rank (PM plan; eval-only) ------
                ## Activated by LIDAR_PICK_TYPE=grid (set globally below). These
                ## configure the (x, y, theta) reference grid used to DECODE each
                ## candidate plan's route and pick the one that actually reaches the
                ## goal cell (maze graph-distance). Read only when pick_type=='grid';
                ## otherwise inert -> default rollout is unchanged.
                args.loc_grid_xy_step = float(os.environ.get('LIDAR_LOC_XYSTEP', '1.0'))
                args.loc_grid_n_theta = int(os.environ.get('LIDAR_LOC_NTHETA', '12'))
                args.loc_grid_sigma = float(os.environ.get('LIDAR_LOC_SIGMA', '1.0'))
                args.loc_n_decode = int(os.environ.get('LIDAR_LOC_SUBSAMPLE', '24'))
                args.loc_grid_cache = os.environ.get(
                    'LIDAR_LOC_CACHE_ROOT', 'data/ogb_maze/lidar_grid')
                ## Goal-cell source for the re-ranker's graph-distance field:
                ##   'true'  = env's true goal (x,y)->cell. DIAGNOSTIC ablation that
                ##             isolates whether SELECTION helps, given a CORRECT target.
                ##             (Re-ranker still only *chooses* among LiDAR plans; the
                ##             model sees no x,y. The true goal is used only to define
                ##             the scoring target cell -- clearly an ablation, not the
                ##             compliant final.)
                ##   'lidar' = localize the goal SCAN on the grid (pure LiDAR, compliant).
                ##             If the goal scan is pose-ambiguous the goal cell may land
                ##             on a look-alike -- that is the effect we are measuring.
                args.loc_goal_mode = os.environ.get('LIDAR_LOC_GOAL_MODE', 'true').lower()

                ## --- PM Stage-3: LiDAR self-localization + goal-arrival termination ---
                ## The ant decodes its own position from LiDAR each step (histogram filter
                ## on the (x,y,theta) grid) and ENDS the run when it enters the goal cell
                ## (scan matches the trained goal scan). LIDAR_ARRIVAL=1 enables it; default
                ## off -> rollout unchanged. Arrival ends the episode but does NOT fake env
                ## success, so ep_srate stays ground-truth and arrival precision is measurable.
                args.lidar_arrival = int(os.environ.get('LIDAR_ARRIVAL', '0'))
                args.lidar_arrival_ends = int(os.environ.get('LIDAR_ARRIVAL_ENDS', '1'))
                args.lidar_arrival_goal_mode = os.environ.get('LIDAR_ARRIVAL_GOAL_MODE', 'lidar').lower()
                args.lidar_arrival_persist = int(os.environ.get('LIDAR_ARRIVAL_PERSIST', '3'))

                ## --- PM GRID NAVIGATOR (likelihood grid drives the route; no retrain) ---
                ## LIDAR_NAV_MODE: '' off (default) | 'bfs' Plan A (the executed plan IS the
                ## grid-routed scan sequence; diffusion sampling bypassed) | 'subgoal' Plan B
                ## (diffusion conditioned on a grid-routed subgoal band ~N cells ahead) |
                ## 'tf_exec' executor diagnostic (plan = a REAL dataset scan sequence).
                args.nav_mode = os.environ.get('LIDAR_NAV_MODE', '').lower()
                ## goal-cell source: 'lidar' localize the goal scan BAND on the grid
                ## (pure LiDAR, compliant) | 'true' env goal cell (diagnostic ablation).
                args.nav_goal_mode = os.environ.get('LIDAR_NAV_GOAL_MODE', 'lidar').lower()
                args.nav_step_wu = float(os.environ.get('LIDAR_NAV_STEP_WU', '0.13'))
                args.nav_yaw_smooth = int(os.environ.get('LIDAR_NAV_YAW_SMOOTH', '31'))
                args.nav_subgoal_cells = float(os.environ.get('LIDAR_NAV_SUBGOAL_CELLS', '10'))
                args.nav_min_conf = float(os.environ.get('LIDAR_NAV_MIN_CONF', '0.02'))
                args.nav_probe = int(os.environ.get('LIDAR_NAV_PROBE', '1'))
                args.nav_goal_hold = int(os.environ.get('LIDAR_NAV_GOAL_HOLD', '40'))
                ## tf_exec: min remaining in-episode frames of the picked real segment
                args.nav_tf_min_frames = int(os.environ.get('LIDAR_NAV_TF_MIN', '120'))
                ## tf_exec heading gate: max |live yaw - segment-start yaw| in degrees
                ## (2026-07-05: unmatched headings put the inv-dyn's obs->goal yaw
                ## delta far outside its k=3-4 training distribution).
                args.nav_tf_max_dyaw = float(os.environ.get('LIDAR_NAV_TF_MAX_DYAW', '60'))
                ## pursuit v2 (2026-07-05 regression fix; see _nav_pick_wp_idx):
                ## cursor PACED +1 frame/step, localization only issues a bounded
                ## (<= nav_corr_max frames/step) correction and only when est_xy
                ## snaps onto the local plan window (<= nav_snap_mj). 0 = pure
                ## time pacing (the 21089 regime).
                args.nav_pursuit = int(os.environ.get('LIDAR_NAV_PURSUIT', '1'))
                args.nav_lookahead = int(os.environ.get('LIDAR_NAV_LOOKAHEAD', '12'))
                args.nav_pursuit_win = int(os.environ.get('LIDAR_NAV_PURSUIT_WIN', '40'))
                args.nav_pursuit_back = int(os.environ.get('LIDAR_NAV_PURSUIT_BACK', '15'))
                args.nav_snap_mj = float(os.environ.get('LIDAR_NAV_SNAP_MJ', '3.0'))
                args.nav_corr_max = int(os.environ.get('LIDAR_NAV_CORR_MAX', '2'))
                ## A2 ribbon source (2026-07-07): 'synth' (default, unchanged) |
                ## 'data' = stitch REAL dataset frames along the BFS route so the
                ## executed goal scans come from the inv-dyn's training manifold
                ## (real poses + gait wobble) instead of ray-cast lattice poses.
                ## Frames from LIDAR_TF_NPZ (default <env>-val.npz; point it at the
                ## full train npz for densest coverage). Dataset xy only picks the
                ## frames (tf_exec standard); the model still sees scans only.
                args.nav_ribbon = os.environ.get('LIDAR_NAV_RIBBON', 'synth').lower()
                args.nav_a2_seg_frames = int(os.environ.get('LIDAR_NAV_A2_SEG', '12'))
                args.nav_a2_snap_wu = float(os.environ.get('LIDAR_NAV_A2_SNAP_WU', '2.0'))
                ## v2b chunk executor (2026-07-07): actions played per model query
                ## for act_chunk inv-dyn runs. 0 = auto (chunk/2). Ignored (inert)
                ## for single-action checkpoints; encoding/chunk themselves are
                ## auto-detected from the inv run's trainer pkl at load time.
                args.act_chunk_exec = int(os.environ.get('LIDAR_ACT_CHUNK_EXEC', '0'))
                ## live-filter motion diffusion (lattice steps/scan; also used by arrival)
                args.loc_motion_sigma = float(os.environ.get('LIDAR_LOC_MOTION_SIGMA', '1.0'))
                ## goal band rendered along the BFS approach corridor instead of the
                ## straight-line most-open-beam walk (train/eval OOD fix; needs K>1).
                args.band_from_route = int(os.environ.get('LIDAR_BAND_FROM_ROUTE', '0'))
                ## goal-contrast guidance weight on the ACTION (2026-07-05): amplify
                ## the inv-dyn's weak goal channel, a = a_self + w*(a_goal - a_self).
                ## 1.0 = off (bit-exact legacy). Probe-measured channel: ~20% of var.
                args.act_cfg_w = float(os.environ.get('LIDAR_ACT_CFG_W', '1.0'))
                ## TOP-K goal-posterior diagnostic (memory 2026-07-08): log the top-K
                ## goal *cells* + scores + where the TRUE cell ranks, one print per ep.
                args.nav_topk = int(os.environ.get('LIDAR_NAV_TOPK', '5'))
                ## verify-and-hop (fix-1): don't trust a self-declared arrival -- match
                ## the live scan to the goal band tail; on mismatch reject the cell and
                ## re-route to the next posterior hypothesis (sequential decoy sweep).
                args.nav_verify_hop = int(os.environ.get('LIDAR_VERIFY_HOP', '0'))
                ## accept tolerance (per-beam RMS world-units) for the live-vs-goal scan
                args.nav_verify_tol = float(os.environ.get('LIDAR_VERIFY_TOL', '1.5'))
                ## max hypothesis visits per episode (budget ~5-6 at ~1300 steps/8000)
                args.nav_verify_hop_max = int(os.environ.get('LIDAR_VERIFY_MAX_HOPS', '6'))
                ## belief-converged gate (fix-2): require this MAP posterior mass before
                ## honoring an arrival (0 = off) -- skips step-61 spawn-phantom arrivals.
                args.nav_verify_min_conf = float(os.environ.get('LIDAR_VERIFY_MIN_CONF', '0.0'))
                ## physical stuck-in-place detection (2026-07-16): independent of
                ## the belief-stuck check above -- see _nav_check_physically_stuck.
                ## 1 = on (default). window/radius tuned from the ep12 x100 repeat
                ## data (~11% of runs spent 30%+ of steps with near-zero movement).
                args.nav_stuck_detect = int(os.environ.get('LIDAR_NAV_STUCK_DETECT', '1'))
                args.nav_stuck_window = int(os.environ.get('LIDAR_NAV_STUCK_WINDOW', '1000'))
                args.nav_stuck_radius = float(os.environ.get('LIDAR_NAV_STUCK_RADIUS', '3.0'))
                ## goal_tol override (2026-07-22): loosen the env's own ground-truth
                ## success radius (see create_env() in ogb_stgl_sml_planner_v1.py).
                ## Unset (default) -> untouched (0.5mj for non-point mazes).
                if 'LIDAR_GOAL_TOL' in os.environ:
                    args.goal_tol_override = float(os.environ['LIDAR_GOAL_TOL'])
                ## fingerprint goal-cell ranking (2026-07-30, validated fix): see
                ## _fingerprint_rank / LIDAR_FP_RANK in ogb_stgl_sml_lidar_planner_v1.py.
                args.nav_fp_rank = int(os.environ.get('LIDAR_FP_RANK', '0'))

                utils.print_color(
                    f'[lidar rollout] is_replan={args.is_replan} '
                    f'lidar_goal_yaw={args.lidar_goal_yaw} '
                    f"repl_every={os.environ.get('LIDAR_REPL_EVERY', '0')} "
                    f"cond_w={os.environ.get('LIDAR_COND_W', '2.0')} "
                    f"pick_type={os.environ.get('LIDAR_PICK_TYPE', 'first')} "
                    f'goal_band_step={args.goal_band_step} tf_band={args.tf_band} '
                    f'tf_band_stride={args.tf_band_stride} '
                    f'loc_goal_mode={getattr(args, "loc_goal_mode", "-")} '
                    f'nav_topk={args.nav_topk} verify_hop={args.nav_verify_hop} '
                    f'verify_tol={args.nav_verify_tol} verify_max_hops={args.nav_verify_hop_max} '
                    f'verify_min_conf={args.nav_verify_min_conf} '
                    f'nav_stuck_detect={args.nav_stuck_detect} '
                    f'nav_stuck_window={args.nav_stuck_window} '
                    f'nav_stuck_radius={args.nav_stuck_radius} '
                    f'goal_tol_override={getattr(args, "goal_tol_override", None)} '
                    f'nav_fp_rank={args.nav_fp_rank}', c='c')
                ## LiDAR inverse-dynamics run dir (must exist on disk). Default path
                ## resolver keys off env+gl_dim and will NOT find this, so set it
                ## explicitly. Confirm the dir name matches your trained run.
                ## --- proprioception fix: obs=[full 29-D ant obs, lidar] (obs_dim 59) ---
                ## The rollout auto-detects the obs_layout from this run's dataset_config,
                ## so no other rollout change is needed. To reproduce the old (car-like,
                ## 0% success) baseline, point this back at '...xyo_g30d_invdyn_h12'.
                ## LIDAR_INV_MODEL_PATH (2026-07-07): switch inv-dyn checkpoints
                ## without a code edit (e.g. the wide-k goal-aware retrain
                ## '..._invdyn_h64_widek'). Default = the h12 run (bit-exact legacy).
                args.inv_model_path = os.environ.get(
                    'LIDAR_INV_MODEL_PATH',
                    'logs/antmaze-giant-stitch-v0/diffusion/og_antM_Gi_lidar30_full_g30d_invdyn_h12')
                args.inv_epoch = os.environ.get('LIDAR_INV_EPOCH', 'latest')

            ## Ant Maze Giant
            elif dfu_ndim == 2:
                ## NOTE: Set the eval start idx, by default starts from 0
                args.ep_st_idx = 0
                repl_wp_cfg = {}
                args.is_replan = 'ada_dist'
                args.n_act_per_waypnt = 1 ## number of actions per waypoint
                # args.ev_cp_infer_t_type = 'same_t' ## ours, faster
                # args.ev_cp_infer_t_type = 'gsc' ## baselines
                
                args.ev_cp_infer_t_type = 'interleave' ## ours
                args.rd_resol = 300 ## website default: 1000
                # args.is_save_pkl = True ## save the plan/rollout trajs to a pkl file

                args.ev_n_comp = 9

                ## high resolution, etc., other eval/render hyperparameters
                # args.rd_resol = 1600
                # args.ep_st_idx = 20
                # args.is_rd_agv = True
                # args.is_use_subgoal_marker = False
                # args.vid_fps = 60
                ## --------------------------------

                args.repl_ada_dist_cfg = dict(
                    max_n_repl=10,
                    # max_n_repl=0, ## 0: no replan
                    thres=4,
                    type='m_2',
                    ada_dist_minus_n_wp=50,
                    cond_2_extra=150, ##
                    n_max_steps=2000, ## 
                )
                args.inv_epoch = int(8e5)
            
            ## Ant Maze Giant Higher Dim
            elif dfu_ndim == 15:
                args.ev_n_comp = 9
                repl_wp_cfg = {}

                args.is_replan = 'ada_dist'
                args.n_act_per_waypnt = 1
                args.ev_cp_infer_t_type = 'interleave'

                args.repl_ada_dist_cfg = dict(
                    max_n_repl=15,
                    thres=4,
                    type='m_2',
                    ada_dist_minus_n_wp=0,
                    cond_2_extra=150, ##
                    n_max_steps=2000, ## 
                    used_idxs=(0,1),
                )
                args.inv_epoch = int(8e5)
            ## Ant Maze Giant
            elif dfu_ndim == 29:
                args.ev_n_comp = 9 # 8
                repl_wp_cfg = {}
                args.is_replan = 'ada_dist'
                args.n_act_per_waypnt = 1 ## important:
                args.ev_cp_infer_t_type = 'interleave'

                args.repl_ada_dist_cfg = dict(
                    max_n_repl=15, ##
                    thres=4,
                    type='m_2',
                    ada_dist_minus_n_wp=0,
                    cond_2_extra=150, ## 100?
                    n_max_steps=2000, ## 
                    used_idxs=(0,1),
                )
                args.inv_epoch = int(8e5)
        
        ## Ant Maze Large Stitch
        elif 'large' in args.dataset:
            if dfu_ndim == 29:
                
                args.ev_cp_infer_t_type = 'interleave' ## gsc / same_t (parallel)
                args.rd_resol = 300 # 1000
                # args.is_save_pkl = True


                repl_wp_cfg = {}
                args.ev_n_comp = 5 ## 6,7 is also fine
                args.is_replan = 'ada_dist'
                args.n_act_per_waypnt = 1

                args.repl_ada_dist_cfg = dict(
                    max_n_repl=10,
                    # max_n_repl=0, ## no replan
                    thres=4, ## 
                    type='m_2',
                    ada_dist_minus_n_wp=50,
                    cond_2_extra=150, ## 100?
                    n_max_steps=1000, ## 
                    used_idxs=(0,1),
                )
                args.inv_epoch = int(8e5)


            ## Ant Maze Large Stitch
            elif dfu_ndim == 15:
                ## ----------------
                args.ev_cp_infer_t_type = 'interleave'

                repl_wp_cfg = {}
                args.ev_n_comp = 6
                args.is_replan = 'ada_dist'
                args.n_act_per_waypnt = 1
                args.repl_ada_dist_cfg = dict(
                    max_n_repl=10,
                    thres=4, 
                    type='m_2',
                    ada_dist_minus_n_wp=50,
                    cond_2_extra=150, ## 100?
                    n_max_steps=1000, ## 
                    used_idxs=(0,1),
                )
                # pdb.set_trace()
                args.inv_epoch = int(8e5)

            ## Ant Maze Large Stitch
            elif dfu_ndim == 2:
                repl_wp_cfg = {}
                
                args.ev_n_comp = 6 ##
                args.ev_cp_infer_t_type = 'interleave'

                args.is_replan = 'ada_dist'
                args.n_act_per_waypnt = 1
                args.repl_ada_dist_cfg = dict(
                    max_n_repl=10,
                    # max_n_repl=0, ## no replan
                    thres=4, 
                    type='m_2',
                    ada_dist_minus_n_wp=50,
                    cond_2_extra=150, ## 
                    n_max_steps=1000, ## 
                )

                args.inv_epoch = int(8e5)


        ## Ant Maze Medium
        elif 'medium' in args.dataset:
            if dfu_ndim == 29:

                ## ---------------------
                repl_wp_cfg = {}
                args.ev_n_comp = 3 ##
                args.is_replan = 'ada_dist'
                args.n_act_per_waypnt = 1
                
                args.repl_ada_dist_cfg = dict(
                    max_n_repl=10,
                    thres=4,
                    type='m_2',
                    ada_dist_minus_n_wp=50,
                    cond_2_extra=150,
                    n_max_steps=2000,
                    used_idxs=(0,1),
                )
                # pdb.set_trace()
                args.inv_epoch = int(8e5)

            ## Ant Maze Medium
            elif dfu_ndim == 15:
                repl_wp_cfg = {}
                args.ev_n_comp = 3 ## 4
                args.is_replan = 'ada_dist'
                args.n_act_per_waypnt = 1

                ## -- failure analysis vis --
                args.ev_cp_infer_t_type = 'interleave'
                # args.rd_resol = 1000
                # args.is_save_pkl = True
                # args.is_rd_agv = True
                args.is_use_subgoal_marker = True
                args.vid_fps = 60
                ## --------------------------


                args.repl_ada_dist_cfg = dict(
                    max_n_repl=10,
                    # max_n_repl=0,
                    thres=4,
                    type='m_2',
                    ada_dist_minus_n_wp=50,
                    cond_2_extra=150,
                    n_max_steps=2000,
                    used_idxs=(0,1),
                )
                args.inv_epoch = int(8e5)

            
            ## Ant Maze Medium
            elif dfu_ndim == 2:

                args.ev_n_comp = 3

                args.ev_cp_infer_t_type = 'interleave'
                # args.rd_resol = 1000
                # args.is_save_pkl = True
                # args.is_rd_agv = True
                # args.is_use_subgoal_marker = True ## False
                # args.vid_fps = 60

                # args.ep_st_idx = 40
                repl_wp_cfg = {}
                args.is_replan = 'ada_dist'
                args.n_act_per_waypnt = 1

                args.repl_ada_dist_cfg = dict(
                    max_n_repl=10,
                    # max_n_repl=0, ## no replan
                    thres=4,
                    type='m_2',
                    ada_dist_minus_n_wp=50,
                    cond_2_extra=150,
                    n_max_steps=1000,
                )
                args.inv_epoch = int(8e5)

    
    ## ----------------------------------------------------------------
    ## ---------------------- AntMaze Explore -------------------------
    ## ----------------------------------------------------------------
    
    ## Ant Maze Explore
    elif 'explore' in  args.dataset.lower() and 'antmaze' in args.dataset.lower():
        ## Explore Large
        if 'large' in args.dataset:
            if dfu_ndim in [29, 15]:
                args.ev_cp_infer_t_type = 'interleave'
                repl_wp_cfg = {}
                args.ev_n_comp = 10
                args.is_replan = 'ada_dist'
                args.n_act_per_waypnt = 1
                args.repl_ada_dist_cfg = dict(
                    max_n_repl=10,
                    thres=2,
                    ada_dist_minus_n_wp=0,
                    type='m_2',
                    cond_2_extra=150,
                    n_max_steps=2000,
                    used_idxs=(0,1),
                )
                args.inv_epoch = int(8e5)

            ## Explore Large
            elif dfu_ndim == 2:
                args.ev_cp_infer_t_type = 'interleave' ## or 'gsc', 'same_t_p'
                # args.rd_resol = 400
                # args.is_save_pkl = True

                repl_wp_cfg = {}
                args.ev_n_comp = 6
                args.is_replan = 'ada_dist'
                args.n_act_per_waypnt = 1
                args.repl_ada_dist_cfg = dict(
                    max_n_repl=10,
                    # max_n_repl=0,
                    thres=2, ##
                    type='m_2',
                    ada_dist_minus_n_wp=0,
                    cond_2_extra=150,
                    n_max_steps=1000,
                )
                args.inv_epoch = int(8e5)

            else:
                raise NotImplementedError
        
        ## Explore Medium
        elif 'medium' in args.dataset:
            if dfu_ndim in [29,15]:
                repl_wp_cfg = {} 
                args.ev_n_comp = 5 ## 4

                args.ev_cp_infer_t_type = 'interleave'
                args.rd_resol = 1000
                args.is_save_pkl = True

                args.is_replan = 'ada_dist'
                args.n_act_per_waypnt = 1

                args.repl_ada_dist_cfg = dict(
                    max_n_repl=10,
                    # max_n_repl=0, ## no replan
                    thres=2, ##
                    type='m_2',
                    ada_dist_minus_n_wp=0,
                    cond_2_extra=150,
                    n_max_steps=1000, ## 2000
                    used_idxs=(0,1),
                )
                args.inv_epoch = int(8e5)
            
            ## Explore Medium
            elif dfu_ndim == 2:
                repl_wp_cfg = {}
                args.ev_n_comp = 5 ## Used

                args.ev_cp_infer_t_type = 'interleave' ## or 'gsc', 'same_t_p'
                args.rd_resol = 1000
                args.is_save_pkl = True

                args.is_replan = 'ada_dist'
                args.n_act_per_waypnt = 1

                args.repl_ada_dist_cfg = dict(
                    max_n_repl=10,
                    # max_n_repl=0, ## replan
                    thres=2,
                    type='m_2',
                    ada_dist_minus_n_wp=0,
                    cond_2_extra=150,
                    n_max_steps=1000,
                )
                args.inv_epoch = int(8e5)


    ## ----------------------------------------------------------------
    ## ---------------------- Humanoid Stitch -------------------------
    ## ----------------------------------------------------------------

    ## Only Implemented 2D For Now
    elif 'humanoid' in args.dataset.lower():
        if 'giant' in args.dataset:
            args.ev_n_comp = 11
            args.inv_epoch = int(16e5)

            args.ev_cp_infer_t_type = 'interleave'
            args.rd_resol = 1000
            args.is_save_pkl = True
            args.ep_st_idx = 80

            repl_wp_cfg = {}
            args.is_replan = 'ada_dist'
            args.n_act_per_waypnt = 1 ## important:
            args.repl_ada_dist_cfg = dict(
                max_n_repl=10,
                thres=10,
                type='m_2',
                ada_dist_minus_n_wp=300,
                # ada_dist_minus_n_wp=400, ## TODO:
                cond_2_extra=150, ## 100?
                n_max_steps=8000, ## NEW Jan 10
            )

        ## Humanoid
        elif 'large' in args.dataset:
            if dfu_ndim == 23:
                raise NotImplementedError
            elif dfu_ndim == 2:
                ## Humanoid Large
                args.ev_n_comp = 6
                args.ev_cp_infer_t_type = 'interleave'
                # args.rd_resol = 400 ## or 1000
                # args.is_save_pkl = True
                # args.ep_st_idx = 80

                ## --------------------
                # args.ev_n_comp = 5 # 5
                # args.is_rd_agv = True
                # args.is_use_subgoal_marker = False
                # args.vid_fps = 120 ## humanoid
                # args.ep_st_idx = 60
                ## -------------------

                args.is_replan = 'ada_dist'
                repl_wp_cfg = {}
                args.n_act_per_waypnt = 1
                
                args.repl_ada_dist_cfg = dict(
                    max_n_repl=10,
                    thres=10,
                    type='m_2',
                    ada_dist_minus_n_wp=300,
                    cond_2_extra=150,
                    n_max_steps=5000, ## default for eval
                )
                args.inv_epoch = int(8e5)
        
        ## Humanoid Medium
        elif 'medium' in args.dataset:
            if dfu_ndim == 23:
                raise NotImplementedError
            elif dfu_ndim == 2:
                ## Humanoid Medium
                args.ev_n_comp = 4

                args.ev_cp_infer_t_type = 'interleave' ## or 'gsc'
                # args.rd_resol = 1000
                # args.is_save_pkl = True
                # args.ep_st_idx = 80

                args.is_replan = 'ada_dist'
                repl_wp_cfg = {}
                
                args.n_act_per_waypnt = 1
                
                args.repl_ada_dist_cfg = dict(
                    max_n_repl=10,
                    thres=10,
                    type='m_2',
                    ada_dist_minus_n_wp=300,
                    cond_2_extra=150,
                    n_max_steps=5000,
                )
            args.inv_epoch = int(8e5)
        else:
            raise NotImplementedError

    elif 'antsoccer' in args.dataset:
        ## Soccer
        if 'arena' in args.dataset:
            ## a 4D diffusion planner: ant x-y, ball x-y 
            if dfu_ndim == 4:
                repl_wp_cfg = {}
                args.ev_n_comp = 5

                args.ev_cp_infer_t_type = 'interleave'
                args.is_use_subgoal_marker = False
                args.ep_st_idx = 60
                ## teaser animation vis
                # args.rd_resol = 1200
                # args.is_rd_agv = True ## render agent view video
                # args.vid_fps = 60

                args.is_replan = 'ada_dist'
                args.n_act_per_waypnt = 1 ##
                
                args.repl_ada_dist_cfg = dict(
                    max_n_repl=10,
                    thres=4,
                    type='m_2',
                    ada_dist_minus_n_wp=50,
                    cond_2_extra=50, ## 
                    n_max_steps=5000, ##
                    used_idxs=(0,1),
                )

                args.inv_epoch = 'latest'
                args.is_inv_train_mode = True

            elif dfu_ndim == 17:
                repl_wp_cfg = {}
                args.ev_n_comp = 5

                args.ev_cp_infer_t_type = 'interleave'
                # args.is_use_subgoal_marker = False
                
                args.ep_st_idx = 60
                ## teaser animation vis
                # args.rd_resol = 1200
                # args.is_rd_agv = True ## render agent view video
                # args.vid_fps = 60

                args.is_replan = 'ada_dist'
                args.n_act_per_waypnt = 1
                args.repl_ada_dist_cfg = dict(
                    max_n_repl=10,
                    thres=4,
                    type='m_2',
                    ada_dist_minus_n_wp=50,
                    cond_2_extra=50, ## 
                    n_max_steps=5000, ## ori default
                    used_idxs=(0,1),
                )

                args.inv_epoch = 'latest'
                args.is_inv_train_mode = True

        ## Soccer
        elif 'medium' in args.dataset:
            if dfu_ndim == 17:
                
                repl_wp_cfg = {}
                args.ev_n_comp = 6
                args.ep_st_idx = 0 ## 20
                args.is_replan = 'ada_dist'
                args.ev_cp_infer_t_type = 'interleave'
                args.rd_resol = 600
                args.is_save_pkl = True
                args.is_use_subgoal_marker = False


                args.n_act_per_waypnt = 2 ## 1 vis

                args.repl_ada_dist_cfg = dict(
                    max_n_repl=10,
                    thres=6,
                    type='m_2',
                    ada_dist_minus_n_wp=0,
                    cond_2_extra=10,
                    n_max_steps=5000,
                    used_idxs=(0,1,),
                )

                args.inv_epoch = 'latest'
                args.is_inv_train_mode = True

            ## ---------------------------------
            ## Soccer Medium
            elif dfu_ndim == 4:
                
                args.ev_cp_infer_t_type = 'interleave'
                args.rd_resol = 1000 # 600 # 1000
                args.is_save_pkl = True
                args.is_use_subgoal_marker = False
                ## for website
                args.ep_st_idx = 20
                args.is_rd_agv = True


                repl_wp_cfg = {}
                ## Jan 10 New Replan Method
                args.ev_n_comp = 8
                args.ev_n_comp = 5
                
                args.ev_n_comp = 6 ## Jan 17
                args.ev_n_comp = 7
                args.ev_n_comp = 8

                args.is_replan = 'ada_dist'
                args.n_act_per_waypnt = 2
                args.n_act_per_waypnt = 1

                args.repl_ada_dist_cfg = dict(
                    max_n_repl=10,
                    thres=4,
                    type='m_2',
                    ada_dist_minus_n_wp=50, ## ?
                    cond_2_extra=50, ## 100?
                    n_max_steps=5000,
                    used_idxs=(0,1,),
                )

                # args.inv_epoch = int(12e5)
                args.inv_epoch = 'latest'
                args.is_inv_train_mode = True

    ## OGBench Point Maze Env
    elif 'pointmaze' in args.dataset.lower():
        
        ## Point Maze Giant
        if 'giant' in args.dataset:
            # args.ev_n_comp = 9
            args.ev_n_comp = 8
            
            # args.ev_cp_infer_t_type = 'same_t' ## parallel
            args.ev_cp_infer_t_type = 'interleave' ## default
            # args.ev_cp_infer_t_type = 'gsc' ## gsc baseline
            # args.ev_cp_infer_t_type = 'same_t_p'
            # args.ev_cp_infer_t_type = 'ar_back' ## backward autoregressive

            # args.rd_resol = 1000
            # args.is_save_pkl = True ## save rollout stats
            
            # args.ep_st_idx = 40 ## starting eval problem idx, default is 0

            ## 
            repl_wp_cfg = {}
            args.is_replan = 'ada_dist'
            args.n_act_per_waypnt = 1
            args.repl_ada_dist_cfg = dict(
                max_n_repl=10,
                # max_n_repl=0, ## no repl abl
                thres=1,
                type='m_2',
                ada_dist_minus_n_wp=10,
                cond_2_extra=150, ##
                n_max_steps=1000, ## 
            )
            args.inv_epoch = int(8e5)

        elif 'large' in args.dataset:
            args.ev_n_comp = 5 ##
            args.ev_n_comp = 6 ## either is fine

            args.ev_cp_infer_t_type = 'interleave'

            repl_wp_cfg = {}
            args.is_replan = 'ada_dist'
            args.n_act_per_waypnt = 1
            args.repl_ada_dist_cfg = dict(
                # max_n_repl=10,
                max_n_repl=0, ## no replan
                thres=1,
                type='m_2',
                ada_dist_minus_n_wp=0,
                cond_2_extra=150, ##
                n_max_steps=1000, ## 
            )
            args.inv_epoch = int(8e5)
            

        elif 'medium' in args.dataset:
            args.ev_n_comp = 3 ## or 4

            args.ev_cp_infer_t_type = 'interleave' ## or 'same_t_p'

            repl_wp_cfg = {}
            args.is_replan = 'ada_dist'
            args.n_act_per_waypnt = 1
            args.repl_ada_dist_cfg = dict(
                # max_n_repl=10,
                max_n_repl=0, ## no replan ablation
                thres=1, ## 4
                type='m_2',
                ada_dist_minus_n_wp=0,
                cond_2_extra=150,
                n_max_steps=1000,
            )
            args.inv_epoch = int(8e5)

    else: 
        raise NotImplementedError
    
    ## Plan-sampling + selection (eval-only; env overrides mirror the LIDAR_* knobs).
    ## LIDAR_PICK_TYPE='goal' turns on the LiDAR-only goal-aware re-rank of the top_n
    ## (pick the candidate whose body arrives closest to the goal). Raise LIDAR_TOP_N /
    ## LIDAR_BSIZE so a good-corridor draw is actually generated AND in the re-rank pool.
    args.b_size_per_prob = int(os.environ.get('LIDAR_BSIZE', '40'))
    args.ev_top_n = int(os.environ.get('LIDAR_TOP_N', '5'))
    args.ev_pick_type = os.environ.get('LIDAR_PICK_TYPE', 'first')
    args.ev_goal_pick_tail = int(os.environ.get('LIDAR_GOAL_PICK_TAIL', '8'))
    args.tjb_blend_type = 'exp'
    args.tjb_exp_beta = 2


    ### Diffusion Sampling Hyper-param
    args.var_temp = 1.0 ## or 0.5
    ## LIDAR_COND_W overrides classifier-free guidance weight (default 2.0).
    ## Only the LiDAR sweep sets it; every other run keeps 2.0 (env var absent).
    args.cond_w = float(os.environ.get('LIDAR_COND_W', '2.0'))
    args.use_ddim = True
    args.ddim_eta = 1.0
    args.ddim_steps = 50
    
    args.repl_wp_cfg = repl_wp_cfg

    latest_e = utils.get_latest_epoch(loadpath)
    # n_e = round(latest_e // 1e5) + 1 # all
    # start_e = 5e5; # 2e5 end_e =
    # depoch_list = np.arange(start_e, int(n_e * 1e5), int(1e5), dtype=np.int32).tolist()

    ## LIDAR_DIFF_EPOCH pins the planner checkpoint step (e.g. an intermediate ckpt
    ## of a still-training run). Empty/absent -> latest (original behavior). The
    ## loaded step is recorded in the rollout json as `epoch_diffusion` -- always
    ## check it matches what you meant to evaluate.
    _de = os.environ.get('LIDAR_DIFF_EPOCH', '').strip()
    depoch_list = [int(float(_de))] if _de else [latest_e,]
    ## depoch_list = [800000,] # 1M
    
    if args.is_replan == 'ada_dist':
        args.env_n_max_steps = args.repl_ada_dist_cfg['n_max_steps']
    else:
        args.env_n_max_steps = None ## use ogb default ??


    ## no-replan (e.g. LiDAR) leaves env_n_max_steps=None; only used for the name
    _ems_k = (args.env_n_max_steps // 1000) if args.env_n_max_steps is not None else 0
    ## many-seed runs (e.g. repeated-episode heatmaps, --pl_seeds "0,1,...,99"):
    ## spelling out every seed overflows the filesystem's filename length limit
    ## (OSError: File name too long). Use a compact range tag instead once the
    ## list is long; the exact-list behavior for the common few-seed case is
    ## unchanged.
    if len(args.pl_seeds) > 8:
        _sd_tag = f"N{len(args.pl_seeds)}-{args.pl_seeds[0]}-{args.pl_seeds[-1]}"
    else:
        _sd_tag = ','.join( [str(sd) for sd in args.pl_seeds] )
    sub_dir = f'{datetime.now().strftime("%y%m%d-%H%M%S-%f")[:-3]}' + \
                        f"-nm{int(args.plan_n_ep)}-ems{_ems_k}k" + \
                        f"-ncp{args.ev_n_comp}" + f"-{args.ev_cp_infer_t_type}"\
                        f"-evSd{_sd_tag}"
    
    # pdb.set_trace()
    ## f'-vt{args.var_temp}'
    if args.is_save_pkl:
        sub_dir += '-pkl'
    if hasattr(args, 'ep_st_idx'):
        sub_dir += f'-st{args.ep_st_idx}'
    if args.is_rd_agv:
        sub_dir += '-agv'

    args.savepath = osp.join(args.savepath, sub_dir)

    result_list = []
    for i in range(len(depoch_list)):
        args_train.diffusion_epoch = depoch_list[i]
        args.diffusion_epoch = depoch_list[i]
        tmp = main( copy.deepcopy(args_train),  copy.deepcopy(args) )
        
        result_list.append(tmp)
    

