"""
LiDAR-aware closed-loop rollout planner for the CompDiffuser giant-antmaze model.

This is the rollout counterpart of the LiDAR *trainer* subclass
(``OgB_Stgl_Sml_Lidar_Trainer_v1``). The base planner
(``OgB_Stgl_Sml_MazeEnvPlanner_V1``) assumes the diffusion model thinks in
``(x, y)``; here the planner observation is a 30-beam LiDAR scan and the
inverse-dynamics model consumes the 34-D ``[x, y, cos(yaw), sin(yaw), lidar]``
composite. We bridge live MuJoCo state -> LiDAR by ray-casting against the maze
with the same simulator used to materialize the training dataset.

Everything about the video machinery (``env.render`` -> ``save_imgs_to_mp4``) is
observation-space agnostic and inherited unchanged: once we emit valid actions,
the ``.mp4`` and ``00_rollout.json`` fall out for free.

The full design rationale lives in ``LIDAR_ROLLOUT_VIDEO_TODO.md`` (repo root).
"""

import os
import time
from collections import deque
from types import SimpleNamespace
import numpy as np

import diffuser.utils as utils
from diffuser.utils import load_config
from diffuser.datasets.normalization import DatasetNormalizer
from diffuser.datasets.lidar.lidar_sim import make_scanner_for_env, obs_to_xy_yaw
from diffuser.datasets.lidar.lidar_grid import load_or_build_grid
from diffuser.datasets.lidar.lidar_localization import GridLocalizer, scan_distance
from diffuser.ogb_task.ogb_maze_v1.ogb_stgl_sml_planner_v1 import (
    OgB_Stgl_Sml_MazeEnvPlanner_V1,
)

## number of leading [x, y, cos(yaw), sin(yaw)] channels in the inv-dyn obs
N_XYO = 4


class OgB_Stgl_Sml_Lidar_MazeEnvPlanner_V1(OgB_Stgl_Sml_MazeEnvPlanner_V1):
    """Closed-loop rollout for the LiDAR giant-antmaze planner."""

    # -- setup -------------------------------------------------------------#
    def setup_load(self, ld_config):
        super().setup_load(ld_config)

        ## The planner observation is the 30-D LiDAR scan, NOT the 2-D (x, y)
        ## placeholder ``obs_select_dim=(0, 1)`` carried by the LiDAR config.
        ## ``dfu_ndim`` controls the width of the fused-plan buffer, so it must
        ## be the true plan dimension (== diffusion observation_dim).
        self.dfu_ndim = self.diffusion.observation_dim

        ## Build the LiDAR simulator from the planner's lidar_config (same geometry
        ## that produced the training scans).
        self.lidar_config = self.args_train.dataset_config['lidar_config']
        self.scanner = make_scanner_for_env(self.env.name, self.lidar_config)

        ## Goal-scan heading: a LiDAR scan is orientation dependent but the eval
        ## problems only give a goal *position*. ``args.lidar_goal_yaw`` is either
        ## a fixed float (default 0.0 -- simplest, deterministic, see §4 of the
        ## TODO) or the string 'goal' to use the goal state's own heading (gl_pos
        ## is a full obs, so it carries a quaternion).
        self.goal_scan_yaw = getattr(self.args, 'lidar_goal_yaw', 0.0)

        ## Multi-scan goal "band": condition the planner on K scans stepping INTO
        ## the goal (K=1 == the original single goal scan). goal_band_step is the
        ## world-unit spacing between consecutive band poses (~1 maze cell = 4 mj).
        self.goal_band_k = int(getattr(self.args, 'goal_band_k', 1))
        self.goal_band_step = float(getattr(self.args, 'goal_band_step', 4.0))

        ## Teacher-forced REAL goal band (diagnostic). When tf_band is set, the goal
        ## band is the TRUE last-K LiDAR scans of a real dataset trajectory ending at
        ## the goal, instead of the synthesized step-back band -- the decisive test of
        ## whether the multi-scan goal CONCEPT works vs. the eval-time synthesis being
        ## out-of-distribution. Data is lazy-loaded on first use.
        self.tf_band = bool(int(getattr(self.args, 'tf_band', 0)))
        ## Frames between consecutive teacher-band scans. S=1 (default) == the last
        ## K *consecutive* frames (<1 cell). S>1 strides the band over SEVERAL cells
        ## of the real approach -> wide-baseline goal context (tests whether position
        ## context, not just a near-goal cluster, routes the ant). Eval-only knob.
        self.tf_band_stride = max(int(getattr(self.args, 'tf_band_stride', 1)), 1)
        self.tf_npz = getattr(self.args, 'tf_npz', '') or \
            os.path.expanduser(f'~/.ogbench/data/{self.env.name}-val.npz')
        self._tf_cache = None
        if self.tf_band:
            utils.print_color(f'[tf_band] ENABLED -- real goal band from {self.tf_npz} '
                              f'(stride={self.tf_band_stride})', c='y')

        ## Goal band rendered along the BFS route into the goal (train/eval OOD fix)
        ## instead of the straight-line most-open-beam walk. Inert when 0.
        self.band_from_route_on = bool(int(getattr(self.args, 'band_from_route', 0)))
        if self.band_from_route_on:
            utils.print_color('[band_from_route] ENABLED -- goal band along the BFS '
                              'approach corridor', c='y')

        ## Goal-contrast guidance on the ACTION (2026-07-05; default 1.0 = off,
        ## bit-exact legacy). The sensitivity probe measured the inv-dyn's goal
        ## channel at ~20% of action variance (stop-vs-go delta 0.11, 24-deg turn
        ## 0.16) -- a whisper against the gait autopilot, hence plan-content-
        ## independent rollouts. w>1 amplifies the goal's contribution CFG-style:
        ##   a = a(s, g_self) + w * (a(s, g_subgoal) - a(s, g_self)),
        ## g_self = the ant's CURRENT scan (the "stay" goal). Pure LiDAR; costs one
        ## extra inv-dyn forward per step. Applied in the base pred_inv controller.
        self._act_cfg_w = float(getattr(self.args, 'act_cfg_w', 1.0) or 1.0)
        if self._act_cfg_w != 1.0:
            utils.print_color(f'[act_cfg] ENABLED w={self._act_cfg_w}', c='y')

        ## The LiDAR scan is always the LAST n_beams dims of the inv-dyn obs, so the
        ## planner plan dim must equal n_beams (works for xyo_lidar AND full_lidar).
        assert self.dfu_ndim == self.n_beams, \
            f'planner LiDAR dim ({self.dfu_ndim}) must equal n_beams ({self.n_beams})'

        utils.print_color(
            f'[lidar planner] dfu_ndim={self.dfu_ndim} goal_scan_yaw={self.goal_scan_yaw} '
            f'scanner={self.scanner.config_dict()}', c='c')

        self._setup_loc_rerank()
        self._setup_arrival()
        self._setup_gridnav()

    # -- localization-guided plan re-rank (PM plan; eval-only, no retrain) --#
    def _setup_loc_rerank(self):
        """Build the (x, y, theta) reference grid + a GridLocalizer and inject a
        re-ranker into the policy. Active ONLY when ``ev_pick_type == 'grid'`` so
        the default rollout is bit-for-bit unchanged. The re-ranker picks the
        candidate plan whose DECODED route (each waypoint scan localized on the
        grid) ends nearest the goal in maze graph-distance -- directly countering
        "the plan routes to a look-alike corridor". Guarded: any failure disables
        it and the policy falls back to consistency-first selection.

        LiDAR compliance: grid scans are rendered from (x, y), exactly as the goal
        scan already is, and the goal *cell* is taken from the eval goal position
        (used only for eval-side path bookkeeping) -- the diffusion model still
        sees only LiDAR scans; the re-ranker only chooses among plans it generated.
        """
        self._loc_rerank_on = (getattr(self.args, 'ev_pick_type', 'first') == 'grid')
        self._rerank_ctx = None
        self._bfs_cache = {}
        ## Goal-cell source for the graph-distance field the re-ranker scores against.
        ## 'true'  -> env's true goal (x,y) cell (DIAGNOSTIC ablation: isolates whether
        ##            SELECTION helps, given a correct target). 'lidar' -> localize the
        ## goal SCAN on the grid (pure LiDAR; exposes goal-scan ambiguity if any).
        self._rerank_goal_mode = str(getattr(self.args, 'loc_goal_mode', 'true')).lower()
        if not self._loc_rerank_on:
            return
        try:
            self.loc_n_decode = int(getattr(self.args, 'loc_n_decode', 24))
            self.loc_grid = load_or_build_grid(
                self.env.name, self.lidar_config,
                xy_step=float(getattr(self.args, 'loc_grid_xy_step', 1.0)),
                n_theta=int(getattr(self.args, 'loc_grid_n_theta', 12)),
                cache_root=getattr(self.args, 'loc_grid_cache', 'data/ogb_maze/lidar_grid'))
            self.loc = GridLocalizer(
                self.loc_grid, sigma=float(getattr(self.args, 'loc_grid_sigma', 1.0)))
            self.policy.pick_type = 'grid'
            self.policy.grid_reranker = self._grid_rerank
            utils.print_color(
                f'[loc rerank] ON: grid {self.loc_grid.n_poses} poses '
                f'(xy_step={self.loc_grid.xy_step}, n_theta={self.loc_grid.n_theta}); '
                f'decode {self.loc_n_decode} waypoints/plan', c='c')
        except Exception as e:  # noqa: BLE001
            self._loc_rerank_on = False
            utils.print_color(
                f'[loc rerank] setup FAILED -> consistency-first selection ({e})', c='y')

    def _bfs_dist_field(self, goal_cell):
        """BFS graph-distance (in maze cells) from ``goal_cell`` over free cells.
        Returns an (H, W) int array; unreachable / wall == large number. Cached."""
        key = tuple(goal_cell)
        if key in self._bfs_cache:
            return self._bfs_cache[key]
        mp = self.scanner.maze_map
        H, W = mp.shape
        BIG = H * W + 1
        dist = np.full((H, W), BIG, dtype=np.int32)
        gi, gj = int(goal_cell[0]), int(goal_cell[1])
        if not (0 <= gi < H and 0 <= gj < W) or mp[gi, gj] == 1:
            self._bfs_cache[key] = dist
            return dist
        dist[gi, gj] = 0
        q = deque([(gi, gj)])
        while q:
            i, j = q.popleft()
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W and mp[ni, nj] == 0 and dist[ni, nj] == BIG:
                    dist[ni, nj] = dist[i, j] + 1
                    q.append((ni, nj))
        self._bfs_cache[key] = dist
        return dist

    def _decode_route(self, scans_seq):
        """Localize a plan's waypoint-scan SEQUENCE on the grid via the motion-history
        filter (resolves the single-scan corridor ambiguity), returning per-step maze
        cells (n, 2). ``scans_seq`` is (n, n_beams), ordered start -> end."""
        r = self.loc.localize_sequence(
            np.asarray(scans_seq, dtype=np.float32),
            motion_sigma_cells=float(getattr(self.args, 'loc_motion_sigma', 2.0)),
            theta_blur=1)
        xy = r['map_xy']
        i, j = self.scanner.world_to_cell(xy[:, 0], xy[:, 1])
        return np.stack([i, j], axis=1).astype(np.int64)

    def _ensure_rerank_ctx(self):
        """Build/refresh the per-episode re-rank context: the goal CELL and the BFS
        graph-distance field candidate routes are scored against. Called at the top of
        ``_grid_rerank`` -- the goal is set on the env before planning, so
        ``env.cur_goal_xy`` is valid here. Rebuilt only when the goal changes.

        Goal-cell source (``self._rerank_goal_mode``):
          * 'true'  -- env's true goal (x, y) -> cell. DIAGNOSTIC ablation that isolates
            whether SELECTION helps, given a correct target. The re-ranker still only
            *chooses* among LiDAR-conditioned plans; the model is fed no (x, y).
          * 'lidar' -- localize the goal SCAN on the grid (pure LiDAR, compliant). If the
            goal scan is pose-ambiguous the goal cell may land on a look-alike -- that is
            the failure mode we are measuring, not a bug. We log how far off it lands.
        """
        gxy = np.asarray(self.env.cur_goal_xy, dtype=np.float64).reshape(-1)
        gx, gy = float(gxy[0]), float(gxy[1])
        prev = self._rerank_ctx
        if prev is not None and prev.get('goal_xy') == (gx, gy):
            return                                    ## same goal this episode -> reuse
        gi, gj = self.scanner.world_to_cell(gx, gy)
        true_cell = (int(gi), int(gj))
        mode = getattr(self, '_rerank_goal_mode', 'true')
        if mode == 'lidar':
            yaw = 0.0 if isinstance(self.goal_scan_yaw, str) else float(self.goal_scan_yaw)
            gscan = self.scanner.scan(gx, gy, yaw).astype(np.float32)
            cell = tuple(int(v) for v in self.loc.localize_single(gscan)['cell'])
        else:
            cell = true_cell
        dist_field = self._bfs_dist_field(cell)
        off = int(dist_field[true_cell[0], true_cell[1]]) if mode == 'lidar' else 0
        self._rerank_ctx = dict(goal_cell=cell, dist_field=dist_field,
                                mode=mode, goal_xy=(gx, gy))
        utils.print_color(
            f'[loc rerank] ctx: goal_cell={cell} mode={mode}'
            + (f' (true={true_cell}; localized goal is {off} cells off)'
               if mode == 'lidar' else f' (true goal cell)'), c='c')

    def _grid_rerank(self, trajs_topn):
        """Pick the candidate plan whose decoded route best reaches the goal.

        trajs_topn: (B, tot_hzn, n_beams) un-normalized scan plans (the top-n).
        Each plan's body (start -> just before the inpainted goal band) is decoded to
        a maze route with the motion-history filter, then scored by how near the route
        ENDS to the goal in maze graph-distance, with tie-breaks rewarding net progress
        and a fully traversable decoded route. Returns the best index (0 on any error).
        """
        try:
            self._ensure_rerank_ctx()             ## build goal cell + BFS field (per ep)
            ctx = self._rerank_ctx
            if ctx is None:
                return 0
            dist_field = ctx['dist_field']
            BIG = int(dist_field.max())
            bl = np.asarray(trajs_topn)
            B, H = bl.shape[0], bl.shape[1]
            K = int(getattr(self, 'goal_band_k', 1))
            n_body = max(H - K, 1)                    ## drop the inpainted goal band
            n_dec = int(min(self.loc_n_decode, n_body))
            wp = np.linspace(0, n_body - 1, n_dec).astype(int)
            scores = np.full(B, -1e18)
            endpts = np.zeros(B, dtype=np.int64)
            for b in range(B):
                cells = self._decode_route(bl[b, wp, :])          ## (n_dec, 2)
                dd = dist_field[cells[:, 0], cells[:, 1]].astype(np.float64)
                reach = float((dd < BIG).mean())     ## fraction of route on free graph
                d_end = float(dd[-1]); d_start = float(dd[0])
                progress = d_start - d_end            ## graph cells closed toward goal
                scores[b] = -d_end + 0.25 * progress + 3.0 * reach
                endpts[b] = int(dd[-1])
            best = int(np.argmax(scores))
            utils.print_color(
                f'[loc rerank] pick idx={best}/{B} decoded-endpoint {endpts[best]} cells '
                f'from goal (first-plan {endpts[0]}; best/worst {endpts.min()}/{endpts.max()})',
                c='c')
            return best
        except Exception as e:  # noqa: BLE001
            utils.print_color(f'[loc rerank] runtime fallback to 0 ({e})', c='y')
            return 0

    # -- PM grid NAVIGATOR: likelihood grid drives the route -----------------#
    def _ensure_grid_loc(self):
        """Build (or reuse) the (x, y, theta) reference grid + GridLocalizer."""
        if getattr(self, 'loc', None) is not None:
            return
        self.loc_grid = load_or_build_grid(
            self.env.name, self.lidar_config,
            xy_step=float(getattr(self.args, 'loc_grid_xy_step', 1.0)),
            n_theta=int(getattr(self.args, 'loc_grid_n_theta', 12)),
            cache_root=getattr(self.args, 'loc_grid_cache', 'data/ogb_maze/lidar_grid'))
        self.loc = GridLocalizer(
            self.loc_grid, sigma=float(getattr(self.args, 'loc_grid_sigma', 1.0)))

    def _setup_gridnav(self):
        """The PM's full loop, closed with the likelihood grid (NO retrain):

        localize self from live scans (histogram filter on the grid) -> localize the
        goal from its scan/band -> BFS route over free cells -> render the scan
        sequence along the route -> the existing waypoint-follower + inv-dyn execute
        it -> the arrival detector ends the run. ``LIDAR_NAV_MODE`` selects:

          ''        (default) off; rollout is bit-for-bit unchanged.
          'bfs'     Plan A: the executed plan IS the grid-routed scan sequence
                    (diffusion sampling is bypassed entirely -- pure navigator).
          'subgoal' Plan B: diffusion still generates the plan, but it is conditioned
                    on a grid-routed SUBGOAL band ~N cells ahead along the BFS route
                    (trained conditioning pattern, receding horizon; n_comp adapts).
          'tf_exec' executor diagnostic: the plan is a REAL dataset trajectory's scan
                    sequence starting near the ant -- does the ant follow real scans?

        Compliance: the executor/model still see only scans; (x, y) is used only to
        render them (the repo standard set by the goal scan + reference grid)."""
        self.nav_mode = str(getattr(self.args, 'nav_mode', '') or '').lower()
        if self.nav_mode in ('', '0', 'off', 'none'):
            self.nav_mode = ''
            return
        assert self.nav_mode in ('bfs', 'subgoal', 'tf_exec'), self.nav_mode
        from diffuser.datasets.lidar.lidar_route_nav import RouteScanSynthesizer
        from diffuser.datasets.lidar.lidar_arrival import LiveLocalizer
        self._ensure_grid_loc()
        self._nav_synth = RouteScanSynthesizer(
            self.scanner,
            step_wu=float(getattr(self.args, 'nav_step_wu', 0.13)),
            yaw_smooth_win=int(getattr(self.args, 'nav_yaw_smooth', 31)))
        self._nav_live = LiveLocalizer(
            self.loc,
            motion_sigma_cells=float(getattr(self.args, 'loc_motion_sigma', 1.0)),
            theta_blur=1)
        self._nav_goal_mode = str(getattr(self.args, 'nav_goal_mode', 'lidar')).lower()
        self._nav_subgoal_cells = float(getattr(self.args, 'nav_subgoal_cells', 10.0))
        self._nav_min_conf = float(getattr(self.args, 'nav_min_conf', 0.02))
        self._nav_probe_on = bool(int(getattr(self.args, 'nav_probe', 1)))
        self._nav_goal_hold = int(getattr(self.args, 'nav_goal_hold', 40))
        self._nav_goal = None
        self._nav_first_plan = True
        self._nav_pending = None
        ## pursuit (closed-loop waypoint progression), REGRESSION-FIXED 2026-07-05.
        ## Run 21129 (0/65 eps left the start corner): the v1 pursuit drove a
        ## MONOTONIC cursor with argmin(|plan_xy - MAP est_xy|) over a 150-frame
        ## window. The MAP est is ~53 mj WRONG at episode start (bottom-left spawn
        ## scan aliases the top-right basin; 16/20 nav_true eps) and teleports
        ## between basins -> the cursor raced to the plan tail within ~10 steps
        ## (verified offline) and could never come back -> subgoal scans cells
        ## ahead -> OOD goal for the k=3-4-trained inv-dyn -> thrash in place.
        ## v2: cursor is PACED (+1 frame/env step = the 21089-proven time-indexed
        ## follower); localization applies only a BOUNDED correction (<= nav_corr_max
        ## frames/step) and only when the est SNAPS onto the local window
        ## (min dist <= nav_snap_mj); phantom ests fail the snap and change nothing.
        self._nav_pursuit_on = bool(int(getattr(self.args, 'nav_pursuit', 1)))
        self._nav_lookahead = int(getattr(self.args, 'nav_lookahead', 12))
        self._nav_win = int(getattr(self.args, 'nav_pursuit_win', 40))
        self._nav_back = int(getattr(self.args, 'nav_pursuit_back', 15))
        self._nav_snap_mj = float(getattr(self.args, 'nav_snap_mj', 3.0))
        self._nav_corr_max = int(getattr(self.args, 'nav_corr_max', 2))
        ## A2 (2026-07-07): ribbon SOURCE. 'synth' (default) = ray-cast scans at
        ## lattice poses (unchanged behavior). 'data' = stitch REAL dataset
        ## frames along the BFS route, so every executed goal scan comes from
        ## the inv-dyn's own training manifold (real gait-wobble poses) instead
        ## of the synthetic corridor-center track. Same diagnostic standard as
        ## tf_exec: dataset (x, y) only PICKS the frames; the model sees scans.
        self._nav_ribbon = str(getattr(self.args, 'nav_ribbon', 'synth')).lower() or 'synth'
        assert self._nav_ribbon in ('synth', 'data'), self._nav_ribbon
        self._nav_a2_seg = int(getattr(self.args, 'nav_a2_seg_frames', 12))
        self._nav_a2_snap_wu = float(getattr(self.args, 'nav_a2_snap_wu', 2.0))
        ## TOP-K goal-posterior diagnostic + verify-and-hop (memory 2026-07-08 fix-1).
        ## Logging (top-K cells + true rank) is always on in lidar goal mode -- it is
        ## the "does the true cell even rank top-K?" diagnostic and costs one print.
        ## Verify-and-hop is opt-in: on a self-declared arrival, compare the LIVE scan
        ## to the goal band's tail scan; a mismatch (> verify_tol per-beam RMS wu)
        ## REJECTS the cell and re-routes to the next-best posterior hypothesis, up to
        ## verify_hop_max visits/episode (budget ~5-6 at ~1300 steps/visit in 8000).
        self._nav_topk = int(getattr(self.args, 'nav_topk', 5))
        self._nav_verify_hop = bool(int(getattr(self.args, 'nav_verify_hop', 0)))
        self._nav_verify_tol = float(getattr(self.args, 'nav_verify_tol', 1.5))
        self._nav_hop_budget = int(getattr(self.args, 'nav_verify_hop_max', 6))
        self._nav_verify_min_conf = float(getattr(self.args, 'nav_verify_min_conf', 0.0))
        self._nav_hop_used = 0
        self._nav_rejected_cells = set()
        self._nav_force_replan = False
        self._nav_plan_xy = None
        self._nav_new_plan = False
        self._nav_plan_start_wp = 0
        self._nav_cursor = 0
        self._nav_prev_t = 0
        ## stuck-belief detection (confident wrong-basin lock observed in the logs)
        self._nav_last_est_cell = None
        self._nav_stuck = 0
        ## stuck-IN-PLACE detection (2026-07-16): the belief-stuck check above only
        ## fires when the localizer's MAP cell doesn't change for 3 replans -- if the
        ## belief keeps flickering between a couple of nearby wrong cells (common
        ## with the aliasing this repo has been chasing), that check resets to 0 and
        ## never fires, even though the ant is physically confined to one area the
        ## whole time (measured: ~11% of repeated-episode runs spent 30%+ of their
        ## steps with near-zero net movement). This tracks the ant's ACTUAL (x, y)
        ## over a rolling window, independent of what the belief thinks, and trips
        ## the SAME soft-reset + probe recovery on physical confinement alone.
        self._nav_stuck_detect = bool(int(getattr(self.args, 'nav_stuck_detect', 1)))
        self._nav_stuck_window = int(getattr(self.args, 'nav_stuck_window', 1000))
        self._nav_stuck_radius = float(getattr(self.args, 'nav_stuck_radius', 3.0))
        self._nav_pos_hist = deque(maxlen=self._nav_stuck_window)
        ## wrap plan generation: 'bfs'/'tf_exec' bypass diffusion sampling entirely;
        ## 'subgoal' adapts n_comp to the route-length then calls the real sampler.
        self._nav_real_gen = self.policy.gen_cond_stgl
        self.policy.gen_cond_stgl = self._nav_gen_cond_stgl
        utils.print_color(
            f'[gridnav] ON mode={self.nav_mode} goal_mode={self._nav_goal_mode} '
            f'step_wu={self._nav_synth.step_wu} subgoal_cells={self._nav_subgoal_cells} '
            f'min_conf={self._nav_min_conf} probe={self._nav_probe_on} '
            f'ribbon={self._nav_ribbon}'
            + (f' (a2_seg={self._nav_a2_seg} a2_snap={self._nav_a2_snap_wu})'
               if self._nav_ribbon == 'data' else '')
            + f' | topk={self._nav_topk} verify_hop={self._nav_verify_hop}'
            + (f' (tol={self._nav_verify_tol} max_hops={self._nav_hop_budget}'
               f' min_conf={self._nav_verify_min_conf})' if self._nav_verify_hop else '')
            + f' | stuck_detect={self._nav_stuck_detect}'
            + (f' (window={self._nav_stuck_window} radius={self._nav_stuck_radius})'
               if self._nav_stuck_detect else ''),
            c='g')

    def _nav_ep_reset(self, gl_pos):
        """Per-episode: reset the live belief and localize the goal (scan/band)."""
        if not self.nav_mode:
            return
        try:
            self._nav_live.reset()
            self._nav_first_plan = True
            self._nav_pending = None
            self._nav_plan_xy = None
            self._nav_new_plan = False
            self._nav_cursor = 0
            self._nav_last_est_cell = None
            self._nav_stuck = 0
            self._nav_pos_hist.clear()
            ## verify-and-hop: fresh hypothesis sweep each episode
            self._nav_hop_used = 0
            self._nav_rejected_cells = set()
            self._nav_force_replan = False
            self._nav_goal = self._nav_localize_goal(gl_pos)
            g = self._nav_goal
            utils.print_color(
                f"[gridnav] ep goal: cell={g['cell']} xy=({g['xy'][0]:.1f},{g['xy'][1]:.1f}) "
                f"mode={self._nav_goal_mode}"
                + (f" (true cell={g['true_cell']}, {g['cell_err']} cells off)"
                   if self._nav_goal_mode == 'lidar' else ''), c='g')
        except Exception as e:  # noqa: BLE001
            self._nav_goal = None
            utils.print_color(f'[gridnav] ep reset FAILED ({e})', c='y')

    def _nav_step_track(self, obs):
        """Per env step: one histogram-filter update from the live scan, plus
        recording the ACTUAL (x, y) for the physical stuck-in-place detector
        (see ``_nav_check_physically_stuck``) -- independent of the belief."""
        if not self.nav_mode:
            return
        try:
            self._nav_live.update(self._obs_to_scan(obs))
            if self._nav_stuck_detect:
                self._nav_pos_hist.append((float(obs[0]), float(obs[1])))
        except Exception as e:  # noqa: BLE001
            utils.print_color(f'[gridnav] track err ({e})', c='y')

    def _nav_localize_goal(self, gl_pos):
        """Goal cell + endpoint. 'true': the env goal (diagnostic ablation).
        'lidar': localize the goal scan BAND on the grid as a short sequence (the
        motion-history filter collapses single-scan look-alikes); endpoint at the
        localized lattice point (finer than the 4-wu cell center).

        Also aggregates the localizer posterior into the TOP-K goal *cells* and
        records where the TRUE cell ranks -- the diagnostic (memory 2026-07-08)
        that decides whether verify-and-hop can reach the true cell -- and stashes
        the ranked hypotheses + the goal band's tail scan for verify-and-hop."""
        gxy_true = np.asarray(gl_pos, dtype=np.float64).reshape(-1)[:2]
        ti, tj = self.scanner.world_to_cell(gxy_true[0], gxy_true[1])
        true_cell = (int(ti), int(tj))
        if self._nav_goal_mode == 'true':
            return dict(cell=true_cell, xy=(float(gxy_true[0]), float(gxy_true[1])),
                        true_cell=true_cell, cell_err=0, hyps=[], true_rank=0,
                        band_tail=None)
        self._nav_goal_band_yaw = None                            # set by _goal_scan[_band]*
        band = self._goal_scan_band(np.asarray(gl_pos))          # (K, nb) pure LiDAR
        if band.shape[0] > 1:
            r = self.loc.localize_sequence(
                band, motion_sigma_cells=max(2.0, float(self.goal_band_step)),
                theta_blur=1)
            belief = r['final_belief']
            gx, gy = float(r['map_xy'][-1][0]), float(r['map_xy'][-1][1])
        else:
            r = self.loc.localize_single(band[0], return_belief=True)
            belief = r['belief']
            gx, gy = float(r['xy'][0]), float(r['xy'][1])

        ## orientation lock (2026-07-30): the goal band's anchor frame was rendered
        ## at a KNOWN real heading (self._nav_goal_band_yaw), even though the model
        ## itself never sees x/y/yaw -- the CODE building the descriptor knows it.
        ## A decoy room that only aliases the goal at a DIFFERENT heading (the
        ## dominant failure mode found in the 2026-07-30 investigation: a room
        ## rotated 90deg from the goal matches almost perfectly, but only at that
        ## specific relative rotation) has no business competing once candidates
        ## are restricted to poses at the SAME orientation the descriptor was
        ## actually recorded at. LIDAR_ORIENT_LOCK_WINDOW widens the allowed band
        ## by that many theta bins each side (0 = exact bin only).
        locked_belief = None
        band_yaw = self._nav_goal_band_yaw
        if band_yaw is not None:
            n_theta = self.loc_grid.n_theta
            bin_width = 2.0 * np.pi / n_theta
            ti0 = int(np.round(((band_yaw - self.loc_grid.theta0) % (2 * np.pi))
                                / bin_width)) % n_theta
            window = int(os.environ.get('LIDAR_ORIENT_LOCK_WINDOW', '0'))
            allowed = sorted({(ti0 + d) % n_theta for d in range(-window, window + 1)})
            mask = np.isin(self.loc.ti, np.asarray(allowed))
            b = np.asarray(belief, dtype=np.float64)
            b_masked = np.where(mask, b, 0.0)
            if b_masked.sum() > 1e-30:
                locked_belief = b_masked / b_masked.sum()
        orient_lock_on = os.environ.get('LIDAR_ORIENT_LOCK', '0') == '1'
        belief_for_rank = belief
        if orient_lock_on and locked_belief is not None:
            belief_for_rank = locked_belief
            k = int(np.argmax(locked_belief))
            gx, gy = float(self.loc.grid.free_xy[k, 0]), float(self.loc.grid.free_xy[k, 1])

        cell = self.loc_grid.cell_of_xy(gx, gy)
        err = abs(cell[0] - true_cell[0]) + abs(cell[1] - true_cell[1])
        ## TOP-K goal-posterior cells (mass aggregated per maze cell) + true rank.
        k_top = int(getattr(self, '_nav_topk', 5))
        ## fingerprint ranking (2026-07-30, validated fix): a full 12-orientation
        ## fingerprint at the goal's own (x,y), compared INDEX-ALIGNED (no rotation
        ## search) against every candidate cell's own 12-orientation fingerprint.
        ## Diagnostics (fp_rank= below) showed this resolves the dominant failure
        ## mode -- 90deg-rotated look-alike rooms winning under the old
        ## sum/peak-of-belief ranking -- putting the true cell at rank 0-2 on all
        ## 11 problems that previously never ranked it inside the top-K at all.
        ## When enabled, this REPLACES hyps/cell/xy (what hop-to-next and the
        ## initial commit actually use), falling back to the belief-based ranking
        ## only if fingerprint ranking itself errors.
        hyps = []
        if os.environ.get('LIDAR_FP_RANK', '0') == '1':
            try:
                fp_k = max(k_top, int(getattr(self, '_nav_hop_budget', k_top)) + 1)
                hyps = self._fingerprint_rank(gl_pos, k=fp_k)
                if hyps:
                    gx, gy = hyps[0]['xy']
                    cell = self.loc_grid.cell_of_xy(gx, gy)
                    err = abs(cell[0] - true_cell[0]) + abs(cell[1] - true_cell[1])
            except Exception as e:  # noqa: BLE001
                hyps = []
                utils.print_color(f'[gridnav/topk] fingerprint ranking failed '
                                  f'({e}); falling back to posterior', c='y')
        if not hyps:
            try:
                hyps = self._goal_posterior_cells(belief_for_rank, k=max(k_top, 1))
            except Exception as e:  # noqa: BLE001  (diagnostic must never break a rollout)
                hyps = []
                utils.print_color(f'[gridnav/topk] posterior aggregation failed ({e})', c='y')
        true_rank = next((n for n, h in enumerate(hyps) if h['cell'] == true_cell), -1)
        ## diagnostic-only (2026-07-30): when the true cell falls outside the
        ## navigation-facing top-K, search deeper into the SAME posterior to see
        ## whether it's just-below-the-cutoff or genuinely absent from the
        ## localizer's ranking entirely. A separate call so `hyps` (what
        ## hop-to-next actually iterates) is never widened -- this must not
        ## change navigation behavior, only the log line.
        deep_rank_str = ''
        rank_search_depth = int(os.environ.get('LIDAR_RANK_SEARCH_DEPTH', '0'))
        if true_rank < 0 and rank_search_depth > 0:
            try:
                deep_hyps = self._goal_posterior_cells(belief, k=rank_search_depth)
                deep_rank = next((n for n, h in enumerate(deep_hyps)
                                   if h['cell'] == true_cell), -1)
                if deep_rank < 0:
                    deep_rank_str = f' deep_rank=not-found/{rank_search_depth}'
                else:
                    true_score = deep_hyps[deep_rank]['score']
                    top_score = deep_hyps[0]['score']
                    deep_rank_str = (f' deep_rank={deep_rank}/{rank_search_depth} '
                                      f'score={true_score:.4f} vs top={top_score:.4f}')
            except Exception as e:  # noqa: BLE001
                deep_rank_str = f' deep_rank_err({e})'
        ## diagnostic-only (2026-07-30): does ranking cells by PEAK pose likelihood
        ## instead of SUMMED column mass put the true cell higher? See docstring on
        ## _goal_posterior_cells. Separate call, `hyps` (what hop-to-next iterates)
        ## is untouched -- this must not change navigation behavior, only the log.
        peak_rank_str = ''
        if os.environ.get('LIDAR_RANK_AGG_COMPARE', '0') == '1':
            try:
                peak_depth = max(k_top, int(getattr(self, '_nav_hop_budget', k_top)) + 1, 20)
                peak_hyps = self._goal_posterior_cells(belief, k=peak_depth, by='peak')
                peak_rank = next((n for n, h in enumerate(peak_hyps)
                                   if h['cell'] == true_cell), -1)
                peak_top_cell = peak_hyps[0]['cell'] if peak_hyps else None
                peak_rank_str = (f' | peak_rank={"not-found" if peak_rank < 0 else peak_rank}'
                                  f'/{peak_depth} peak_argmax={peak_top_cell}')
            except Exception as e:  # noqa: BLE001
                peak_rank_str = f' | peak_rank_err({e})'
        ## diagnostic-only (2026-07-30): does restricting candidates to the goal
        ## band's OWN recorded orientation (see block above) put the true cell
        ## higher, even when LIDAR_ORIENT_LOCK isn't the active navigation mode?
        orient_rank_str = ''
        if os.environ.get('LIDAR_RANK_AGG_COMPARE', '0') == '1':
            if locked_belief is None:
                orient_rank_str = ' | orient_rank=no-yaw'
            else:
                try:
                    orient_depth = max(k_top, int(getattr(self, '_nav_hop_budget', k_top)) + 1, 20)
                    orient_hyps = self._goal_posterior_cells(locked_belief, k=orient_depth)
                    orient_rank = next((n for n, h in enumerate(orient_hyps)
                                         if h['cell'] == true_cell), -1)
                    orient_top_cell = orient_hyps[0]['cell'] if orient_hyps else None
                    orient_rank_str = (
                        f' | orient_rank={"not-found" if orient_rank < 0 else orient_rank}'
                        f'/{orient_depth} orient_argmax={orient_top_cell}')
                except Exception as e:  # noqa: BLE001
                    orient_rank_str = f' | orient_rank_err({e})'
        ## diagnostic-only (2026-07-30, user's proposal): stationary 12-heading
        ## fingerprint at the goal's own (x,y), index-aligned against every
        ## candidate's own 12-heading fingerprint (see _fingerprint_rank).
        fp_rank_str = ''
        if os.environ.get('LIDAR_RANK_AGG_COMPARE', '0') == '1':
            try:
                fp_depth = max(k_top, int(getattr(self, '_nav_hop_budget', k_top)) + 1, 20)
                fp_hyps = self._fingerprint_rank(gl_pos, k=fp_depth)
                fp_rank = next((n for n, h in enumerate(fp_hyps)
                                 if h['cell'] == true_cell), -1)
                fp_top_cell = fp_hyps[0]['cell'] if fp_hyps else None
                fp_rank_str = (f' | fp_rank={"not-found" if fp_rank < 0 else fp_rank}'
                                f'/{fp_depth} fp_argmax={fp_top_cell}')
            except Exception as e:  # noqa: BLE001
                fp_rank_str = f' | fp_rank_err({e})'
        if hyps:
            top_str = ', '.join(f"{h['cell']}:{h['score']:.3f}" for h in hyps[:k_top])
            utils.print_color(
                f"[gridnav/topk] goal-posterior top{k_top}: {top_str} | true={true_cell} "
                f"rank={'>K' if true_rank < 0 else true_rank}{deep_rank_str} "
                f"(argmax {cell}, {err} cells off){peak_rank_str}{orient_rank_str}{fp_rank_str}", c='c')
        return dict(cell=cell, xy=(gx, gy), true_cell=true_cell, cell_err=int(err),
                    hyps=hyps, true_rank=int(true_rank),
                    band_tail=np.asarray(band[-1], dtype=np.float32).copy())

    def _goal_posterior_cells(self, belief_flat, k=5, by='sum'):
        """Aggregate a per-free-pose posterior ``belief_flat`` (M,) into per-maze-cell
        scores -- the goal-CELL posterior both the navigator and the arrival detector
        act on.

        ``by='sum'`` (default, original behavior): mass is SUMMED over every theta bin
        and every lattice pose inside a cell. This conflates "plausible at many
        orientations, weakly" with "one sharp, sequence-consistent peak at a single
        orientation" -- a cell whose true match requires one specific heading (the
        common case for a room that only aliases the goal when *rotated* some exact
        amount, see 2026-07-30 aliasing investigation) gets no credit for how sharp
        that peak is, only for how much total mass happens to land in its column.

        ``by='peak'``: score is the single highest-belief POSE in the cell (the same
        pose ``localize_sequence`` already treats as MAP-consistent across the whole
        motion-filtered scan sequence) instead of the column sum. Both scores are
        always attached to every entry (``score_sum``/``score_peak``) regardless of
        ``by``, so callers can compare without a second aggregation pass.

        Each entry: ``dict(cell=(i,j), xy=(x,y), score, score_sum, score_peak)`` where
        ``xy`` is the highest-belief lattice pose in the cell (a routable, sub-cell
        endpoint). Sorted by ``score`` desc and kept a little deeper than ``k`` so
        verify-and-hop always has fallbacks."""
        grid = self.loc.grid
        b = np.asarray(belief_flat, dtype=np.float64).reshape(-1)
        b = b / max(b.sum(), 1e-30)
        fxy = np.asarray(grid.free_xy, dtype=np.float64)          # (M, 2)
        ci, cj = self.scanner.world_to_cell(fxy[:, 0], fxy[:, 1])  # (M,), (M,)
        ci = np.asarray(ci, dtype=np.int64).reshape(-1) + 1000     # offset: robust to <0
        cj = np.asarray(cj, dtype=np.int64).reshape(-1) + 1000
        keys = ci * 100000 + cj
        uniq, inv = np.unique(keys, return_inverse=True)
        mass = np.zeros(uniq.shape[0], dtype=np.float64)
        np.add.at(mass, inv, b)
        ## representative pose per cell = its highest-belief lattice pose
        best_pose = np.full(uniq.shape[0], -1, dtype=np.int64)
        order = np.argsort(b, kind='stable')                      # ascending -> last=max
        best_pose[inv[order]] = order
        peak = np.where(best_pose >= 0, b[np.clip(best_pose, 0, None)], 0.0)
        score_arr = peak if by == 'peak' else mass
        n_keep = int(max(int(k), int(getattr(self, '_nav_hop_budget', k)) + 1))
        top = np.argsort(score_arr)[::-1][:n_keep]
        out = []
        for g in top:
            key = int(uniq[g])
            i_cell, j_cell = key // 100000 - 1000, key % 100000 - 1000
            p = int(best_pose[g])
            out.append(dict(cell=(int(i_cell), int(j_cell)),
                            xy=(float(fxy[p, 0]), float(fxy[p, 1])),
                            score=float(score_arr[g]),
                            score_sum=float(mass[g]), score_peak=float(peak[g])))
        return out

    def _fingerprint_rank(self, gl_pos, k=200):
        """(2026-07-30, user's proposal) Rank candidate maze cells by a full
        12-orientation LiDAR 'fingerprint' comparison, INDEX-ALIGNED -- no
        rotation search at all, unlike everything above.

        Renders 12 fresh scans at the goal's own (x, y), one per theta bin
        (0deg, 30deg, ... 330deg) -- a stationary rotational signature of the
        goal location itself, computed purely from known maze geometry (the
        model still never sees x, y; this is the same kind of geometry-only
        synthesis the non-teacher-forced single-scan path already does). Every
        grid lattice point already has its own such 12-scan fingerprint
        precomputed in ``grid.scans_grid`` (n_row, n_col, n_theta, n_beams).
        Compares fingerprint[t] to fingerprint[t] for every t in 0..11 (never
        fingerprint[t] to fingerprint[t'] for t' != t) and sums the 12 per-beam
        RMS diffs -- so a room that only resembles the goal after a 90deg spin
        scores 9 of its 12 headings as bad mismatches instead of hiding behind
        its one best angle. Lower total = better match; unlike
        ``_goal_posterior_cells`` (higher score first) this is ascending.

        Returns up to ``k`` entries, sorted ascending by ``score`` (the summed
        12-heading diff), each cell appearing once (its own best lattice
        point)."""
        grid = self.loc_grid
        gx, gy = float(gl_pos[0]), float(gl_pos[1])
        goal_fp = np.stack([self.scanner.scan(gx, gy, float(th)).astype(np.float32)
                             for th in grid.theta_vals])              # (n_theta, nb)
        sg = np.asarray(grid.scans_grid, dtype=np.float32)            # (n_row,n_col,n_theta,nb)
        with np.errstate(invalid='ignore'):
            diff = np.sqrt(np.nanmean((sg - goal_fp[None, None]) ** 2, axis=-1))  # (row,col,theta)
            total = np.nansum(diff, axis=-1)                          # (n_row, n_col)
        ## NOTE: np.nansum of an all-NaN slice silently returns 0.0 (not NaN), so
        ## wall (row,col) cells -- every theta NaN -- would otherwise look like a
        ## perfect 0.000 match instead of being excluded. Use the grid's own
        ## free/wall mask as ground truth, not NaN-ness of `total` (2026-07-30 bug
        ## caught in local sanity check before the cluster run).
        free = np.asarray(grid.free_mask_xy, dtype=bool)
        rows, cols = np.nonzero(free)
        if rows.size == 0:
            return []
        xs = grid.x_vals[cols].astype(np.float64)
        ys = grid.y_vals[rows].astype(np.float64)
        ci, cj = self.scanner.world_to_cell(xs, ys)
        keys = (np.asarray(ci, dtype=np.int64) + 1000) * 100000 + (np.asarray(cj, dtype=np.int64) + 1000)
        scores = total[rows, cols]
        order = np.argsort(scores)                                   # ascending: best first
        seen = set()
        out = []
        for idx in order:
            key = int(keys[idx])
            if key in seen:
                continue
            seen.add(key)
            i_cell, j_cell = key // 100000 - 1000, key % 100000 - 1000
            out.append(dict(cell=(int(i_cell), int(j_cell)),
                            xy=(float(xs[idx]), float(ys[idx])),
                            score=float(scores[idx])))
            if len(out) >= k:
                break
        return out

    def _nav_est_pose(self):
        """Current MAP pose from the live belief, refreshed with the current scan."""
        st = self._nav_live.update(self._obs_to_scan(self._nav_obs_cur))
        return st

    def _lidar_route_deviation(self):
        """Distance between the ant's own scan-based live position estimate
        (``_nav_live.est_xy``) and where the CURRENT route says it should be
        right now -- the deviation signal for the 'lidar_dist_check' replan
        trigger.

        Deliberately never touches ground-truth (x, y): both sides come from
        LiDAR-derived state the system already tracks for pursuit.

        SNAP-GATED (2026-08-17 fix): reuses the exact same sanity check
        ``_nav_pick_wp_idx`` already applies before trusting ``est_xy`` for
        cursor correction -- search a local window around the cursor
        (``_nav_back``/``_nav_win`` frames each way) for the nearest route
        point to the live estimate, and only trust the estimate if that
        nearest point is within ``_nav_snap_mj``. Without this, a phantom
        estimate (a documented, systematic-at-episode-start glitch already
        worked around in ``_nav_pick_wp_idx`` -- see its docstring, ~50+ mj
        bogus jumps) reads as a huge, spurious deviation and burns a replan
        on a false alarm instead of a real course correction. When the
        estimate fails the snap gate, this returns None (no signal this
        step) rather than a number that would trigger on the glitch."""
        exy = getattr(self._nav_live, 'est_xy', None)
        xy = getattr(self, '_nav_plan_xy', None)
        cursor = getattr(self, '_nav_cursor', None)
        if exy is None or xy is None or cursor is None or len(xy) == 0:
            return None
        cursor = int(cursor)
        n = len(xy)
        back = int(getattr(self, '_nav_back', 0))
        win = int(getattr(self, '_nav_win', 0))
        j0 = max(cursor - back, 0)
        j1 = min(cursor + win, n)
        if j1 <= j0:
            return None
        exy = np.asarray(exy, dtype=np.float64)
        window_xy = np.asarray(xy[j0:j1], dtype=np.float64)
        d = np.linalg.norm(window_xy - exy[None, :], axis=1)
        jm = int(np.argmin(d))
        snap_mj = float(getattr(self, '_nav_snap_mj', 0.0))
        if float(d[jm]) > snap_mj:
            return None  ## phantom reading -- not trustworthy, skip this step
        ## trustworthy: distance from the ant's real (estimated) position to
        ## where the route currently expects it to be, at the cursor
        idx = int(np.clip(cursor, 0, n - 1))
        return float(np.linalg.norm(exy - np.asarray(xy[idx], dtype=np.float64)))

    def _nav_proprio_yaw(self, st=None):
        """The ant's own heading from the live obs quaternion (obs[3:7]).

        Orientation only -- no (x, y) -- so it stays in-spec. Used for the
        ribbon's yaw ramp instead of the localizer's est_theta: at episode
        start the belief theta is as phantom as its xy, and a ribbon whose
        first goal scans are rotated >> gait-wobble vs the live scan is OOD
        for the k=3-4-trained inv-dyn (training never contained large
        obs->goal yaw deltas). Falls back to est_theta."""
        try:
            _, yaw = obs_to_xy_yaw(np.asarray(self._nav_obs_cur)[None], has_quat=True)
            return float(yaw[0])
        except Exception:  # noqa: BLE001
            return float(st['est_theta']) if st is not None else 0.0

    def _nav_route_from_est(self, st):
        """BFS route from the estimated pose to the localized goal -> scan plan."""
        return self._nav_synth.synthesize(
            start_xy=np.asarray(st['est_xy'], dtype=np.float64),
            goal_xy=np.asarray(self._nav_goal['xy'], dtype=np.float64),
            start_yaw=self._nav_proprio_yaw(st),
            goal_hold=self._nav_goal_hold)

    def _nav_check_physically_stuck(self):
        """True if the ant's ACTUAL (x, y) -- not the belief's estimate -- has
        stayed within ``_nav_stuck_radius`` of its own centroid for the full
        ``_nav_stuck_window`` most recent steps. Checked at each plan-build (see
        ``_nav_build_plan_bfs``) alongside the belief-stuck check, but on
        completely independent evidence: catches the case where the localizer's
        MAP cell keeps flickering between a couple of nearby wrong cells (never
        satisfying the belief-stuck condition) while the ant is still physically
        confined to one area the whole time."""
        if not self._nav_stuck_detect or len(self._nav_pos_hist) < self._nav_stuck_window:
            return False
        pts = np.asarray(self._nav_pos_hist, dtype=np.float64)
        centroid = pts.mean(axis=0)
        max_r = float(np.max(np.linalg.norm(pts - centroid, axis=1)))
        return max_r <= self._nav_stuck_radius

    def _nav_build_plan_bfs(self):
        """Plan A: the executed plan is the grid-routed scan sequence itself."""
        st = self._nav_est_pose()
        ## stuck-belief detection: the filter can lock CONFIDENTLY onto a distant
        ## look-alike basin (observed: same wrong cell for 8+ replans, conf 0.3-0.7).
        ## If the MAP cell hasn't moved for 3 consecutive replans, soft-reset the
        ## belief (uniform prior + current scan) and PROBE to break the symmetry
        ## with fresh motion parallax.
        est_cell = self.loc_grid.cell_of_xy(st['est_xy'][0], st['est_xy'][1])
        force_probe = False
        if est_cell == self._nav_last_est_cell:
            self._nav_stuck += 1
        else:
            self._nav_stuck = 0
        self._nav_last_est_cell = est_cell
        if self._nav_stuck >= 2:
            utils.print_color(
                f'[gridnav/bfs] belief STUCK at cell {est_cell} '
                f'({self._nav_stuck + 1} replans) -> soft reset + probe', c='y')
            self._nav_live.reset()
            st = self._nav_est_pose()                            # fresh single-scan MAP
            self._nav_stuck = 0
            force_probe = True
        ## stuck-IN-PLACE detection: independent of the belief above -- the ant's
        ## ACTUAL (x, y) confined to one small area for a long window, whatever
        ## the (possibly flickering) belief thinks. Same recovery: soft reset +
        ## probe, so a fresh scan/motion-parallax cycle gets a chance to escape
        ## the area instead of continuing to re-route from a stale/local estimate.
        if not force_probe and self._nav_check_physically_stuck():
            pts = np.asarray(self._nav_pos_hist, dtype=np.float64)
            centroid = pts.mean(axis=0)
            utils.print_color(
                f'[gridnav/bfs] PHYSICALLY STUCK near ({centroid[0]:.1f},{centroid[1]:.1f}) '
                f'(< {self._nav_stuck_radius}mj for {len(self._nav_pos_hist)} steps) '
                f'-> soft reset + probe', c='y')
            self._nav_live.reset()
            st = self._nav_est_pose()                            # fresh single-scan MAP
            self._nav_pos_hist.clear()                           # don't re-trigger next check
            force_probe = True
        txy = self.env.get_xy()                                  # logging only
        est_err = float(np.linalg.norm(np.asarray(st['est_xy']) - np.asarray(txy)))
        use_probe = force_probe or (self._nav_first_plan and self._nav_probe_on
                                    and st['conf'] < self._nav_min_conf)
        route = None
        if not use_probe and self._nav_goal is not None:
            route = self._nav_route_from_est(st)
        if route is None:                                        # cold start / no path
            scan_now = self._obs_to_scan(self._nav_obs_cur)
            route = self._nav_synth.probe(
                np.asarray(st['est_xy'], dtype=np.float64), self._nav_proprio_yaw(st),
                scan_now, goal_hold=0)
            tag = 'PROBE'
        else:
            tag = f"route {len(route['cells'])} cells"
        ## A2: re-source the ribbon from REAL dataset frames (probe routes have
        ## cells=None and stay synthetic -- they predate a usable belief anyway).
        if (self._nav_ribbon == 'data' and route.get('cells') is not None
                and len(route['xy']) >= 2):
            try:
                route = self._route_to_data_ribbon(route)
                tag += ' [A2 data-ribbon]'
            except Exception as e:  # noqa: BLE001
                utils.print_color(f'[gridnav/A2] data ribbon FAILED -> synth ({e})', c='y')
        self._nav_first_plan = False
        ## pursuit bookkeeping: the executor follows THESE positions
        self._nav_plan_xy = np.asarray(route['xy'], dtype=np.float64)
        self._nav_new_plan = True
        utils.print_color(
            f"[gridnav/bfs] {tag}: est=({st['est_xy'][0]:.1f},{st['est_xy'][1]:.1f}) "
            f"conf={st['conf']:.3f} est_err={est_err:.1f}mj "
            f"plan={route['scans'].shape[0]} frames", c='g')
        return route['scans']

    def _nav_pick_wp_idx(self, wp_idx_time, fused_len):
        """Closed-loop waypoint progression v2 (pure LiDAR), fixed 2026-07-05.

        v1 regression (run 21129, all 65 eps pinned at start): the cursor was
        argmin(|plan_xy - MAP est_xy|) over a forward-only window, monotonic.
        One phantom-basin est (systematic at ep start, est_err ~53 mj) raced it
        to the plan tail in ~10 steps, permanently -> subgoal scan cells ahead
        -> OOD goal for the k=3-4-trained inv-dyn -> thrash in place. This also
        poisoned tf_exec, so the executor rung never actually re-ran.

        v2: baseline pacing = +1 frame per env step (the time-indexed follower
        that navigated 8 cells in 21089). The localizer only issues a bounded
        correction (clip to +-nav_corr_max*dt frames) and only when its est_xy
        SNAPS onto the local plan window (min dist <= nav_snap_mj, searched
        nav_pursuit_back back / nav_pursuit_win ahead). A ~50 mj phantom est
        fails the snap -> pacing untouched; sustained, consistent evidence can
        still hold the cursor back for a slow ant (the original desync fix)."""
        if (not getattr(self, '_nav_pursuit_on', False)
                or self.nav_mode not in ('bfs', 'tf_exec')
                or getattr(self, '_nav_plan_xy', None) is None):
            return wp_idx_time
        try:
            if self._nav_new_plan:
                self._nav_new_plan = False
                self._nav_plan_start_wp = wp_idx_time            # plan starts at this wp
                self._nav_cursor = 0
                self._nav_prev_t = int(wp_idx_time)
            xy = self._nav_plan_xy
            n = len(xy)
            dt = max(int(wp_idx_time) - self._nav_prev_t, 0)
            self._nav_prev_t = int(wp_idx_time)
            ## baseline: consume the ribbon at plan speed (dataset-speed pacing)
            self._nav_cursor = min(self._nav_cursor + dt, n - 1)
            ## snap-gated, rate-limited localization correction
            snapped = False
            exy = getattr(self._nav_live, 'est_xy', None)
            if exy is not None and dt > 0:
                j0 = max(self._nav_cursor - self._nav_back, 0)
                j1 = min(self._nav_cursor + self._nav_win, n)
                if j1 > j0:
                    d = np.linalg.norm(
                        xy[j0:j1] - np.asarray(exy, dtype=np.float64)[None, :], axis=1)
                    jm = int(np.argmin(d))
                    if float(d[jm]) <= self._nav_snap_mj:
                        snapped = True
                        corr = int(np.clip((j0 + jm) - self._nav_cursor,
                                           -self._nav_corr_max * dt,
                                           self._nav_corr_max * dt))
                        self._nav_cursor = int(np.clip(self._nav_cursor + corr, 0, n - 1))
            if wp_idx_time > 0 and wp_idx_time % 250 == 0:
                utils.print_color(
                    f'[pursuit] t={wp_idx_time} cursor={self._nav_cursor}/{n - 1} '
                    f'snap={snapped}', c='c')
            tgt = min(self._nav_cursor + self._nav_lookahead, n - 1)
            return int(min(self._nav_plan_start_wp + tgt, fused_len - 1))
        except Exception:  # noqa: BLE001
            return wp_idx_time

    # -- A2: real-dataset-scan ribbon along the BFS route (2026-07-07) ------#
    def _route_to_data_ribbon(self, route):
        """Re-source the routed ribbon from REAL dataset frames (A2, no retrain).

        WHY: the goal-sensitivity probe showed the h12 inv-dyn responds to real
        k-ahead scans (shuffled delta 0.25, MSE 3x) yet every closed-loop arm was
        plan-content-independent. The executed subgoals are the one input that
        never came from the training manifold: the synthetic ribbon ray-casts at
        LATTICE poses (corridor-center track, smoothed tangent yaw, no gait
        wobble). This swaps them for the nearest real frames: walk the route,
        and at each stretch pick a dataset frame close in xy AND heading-aligned
        with the route tangent (strict pass ``nav_tf_max_dyaw``, then relaxed),
        then keep its next ``nav_a2_seg_frames`` in-episode frames (a real
        micro-segment with true pose dynamics), project the segment end back
        onto the route, repeat. Scans are re-rendered with the training scanner
        (identical to the train-cache law), so the executor consumes exactly
        train-distribution goals at dataset speed (~0.13 wu/frame == step_wu ->
        the pursuit's +1 frame/step pacing still applies).

        Compliance: dataset (x, y, yaw) only SELECTS frames (the tf_exec
        standard); the model still sees scans only. route = dict(scans, xy,
        yaw, cells); returns the same structure. Raises on failure (caller
        falls back to the synthetic ribbon).
        """
        data = self._load_tf_data()
        ep_id = data['ep_id']
        if 'ep_end' not in data:                     # last in-episode index per frame
            N = len(ep_id)
            ep_end = np.empty(N, dtype=np.int64)
            last = N - 1
            for i in range(N - 1, -1, -1):
                if i < N - 1 and ep_id[i] != ep_id[i + 1]:
                    last = i
                ep_end[i] = last
            data['ep_end'] = ep_end
        ep_end = data['ep_end']
        if 'yaw' not in data:                        # per-frame heading (proprio quat)
            data['yaw'] = obs_to_xy_yaw(data['obs'], has_quat=True)[1].astype(np.float32)
        dyaw_all = data['yaw']
        dxy = data['xy']
        if 'cell_buckets' not in data:               # cell -> frame idxs (fast NN, no scipy)
            ci, cj = self.scanner.world_to_cell(dxy[:, 0], dxy[:, 1])
            buck = {}
            for n_i, key in enumerate(zip(np.asarray(ci).tolist(), np.asarray(cj).tolist())):
                buck.setdefault(key, []).append(n_i)
            data['cell_buckets'] = {k: np.asarray(v, dtype=np.int64) for k, v in buck.items()}
        buckets = data['cell_buckets']

        def _cands(pt, k_keep=64):
            """Frame idxs near pt (3x3 cell neighborhood), nearest-first."""
            pi, pj = self.scanner.world_to_cell(np.asarray([pt[0]]), np.asarray([pt[1]]))
            pools = [buckets[key] for di in (-1, 0, 1) for dj in (-1, 0, 1)
                     if (key := (int(pi[0]) + di, int(pj[0]) + dj)) in buckets]
            if not pools:
                return np.empty(0, np.int64), np.empty(0, np.float64)
            idxs = np.concatenate(pools)
            d = np.linalg.norm(dxy[idxs].astype(np.float64) - pt[None, :], axis=1)
            o = np.argsort(d)[:k_keep]
            return idxs[o], d[o]

        ## strip the goal_hold tail (repeated final pose); re-appended at the end
        rxy = np.asarray(route['xy'], dtype=np.float64)
        ryaw = np.asarray(route['yaw'], dtype=np.float64)
        n_r = len(rxy)
        while n_r > 1 and float(np.linalg.norm(rxy[n_r - 1] - rxy[n_r - 2])) < 1e-9:
            n_r -= 1
        max_dyaw = float(getattr(self.args, 'nav_tf_max_dyaw', 60.0)) * np.pi / 180.0
        seg_len = max(int(self._nav_a2_seg), 1)
        snap_wu = float(self._nav_a2_snap_wu)

        rows = []                    # (is_data, dataset idx | fill row) in ribbon order
        fill_xy, fill_yaw = [], []   # synthetic fill rows (kept from the lattice)
        p, n_seg, n_relax, n_fill = 0, 0, 0, 0
        snap_ds = []
        guard = 0
        while p < n_r - 1 and guard < 4 * n_r + 64:
            guard += 1
            ci, cd = _cands(rxy[p])
            pick = -1
            if len(ci) > 0:
                dy = np.abs(np.arctan2(np.sin(dyaw_all[ci] - ryaw[p]),
                                       np.cos(dyaw_all[ci] - ryaw[p])))
                for pass_dyaw in (max_dyaw, np.inf):     # strict heading, then relaxed
                    ok = np.where((cd <= snap_wu) & (dy <= pass_dyaw))[0]
                    if len(ok) > 0:
                        pick = int(ci[ok[0]])
                        snap_ds.append(float(cd[ok[0]]))
                        n_relax += int(pass_dyaw is np.inf)
                        break
            if pick < 0:
                ## no real frame within snap_wu here: keep the synthetic lattice
                ## frames for this stretch (rare with a full dataset; logged)
                q = min(p + seg_len, n_r - 1)
                for t in range(p, q):
                    rows.append((False, len(fill_xy)))
                    fill_xy.append(rxy[t]); fill_yaw.append(ryaw[t])
                n_fill += q - p
                p = q
                continue
            seg = np.arange(pick, int(min(pick + seg_len - 1, ep_end[pick])) + 1,
                            dtype=np.int64)
            ## trim the snippet where it leaves the route corridor (junction turns)
            w = rxy[p:min(p + 4 * seg_len, n_r)]
            dmat = np.linalg.norm(dxy[seg].astype(np.float64)[:, None, :]
                                  - w[None, :, :], axis=2)
            off = np.where(dmat.min(axis=1) > 1.5 * snap_wu)[0]
            if len(off) > 0:
                seg = seg[:max(int(off[0]), 1)]
                dmat = dmat[:len(seg)]
            rows.extend((True, int(v)) for v in seg)
            n_seg += 1
            p += max(int(dmat[-1].argmin()), 1)          # project seg end onto route

        if not rows:
            raise RuntimeError('A2 produced an empty ribbon')
        ## assemble: one scan_obs call for the real frames, one scan_batch for fill
        d_idx = np.asarray([i for is_d, i in rows if is_d], dtype=np.int64)
        scans_d = (self.scanner.scan_obs(data['obs'][d_idx], has_quat=True)
                   .astype(np.float32) if len(d_idx) else
                   np.zeros((0, self.n_beams), dtype=np.float32))
        if fill_xy:
            scans_f = self.scanner.scan_batch(
                np.asarray(fill_xy, dtype=np.float64),
                np.asarray(fill_yaw, dtype=np.float64)).astype(np.float32)
        T = len(rows)
        scans = np.empty((T, scans_d.shape[1] if len(d_idx) else scans_f.shape[1]),
                         dtype=np.float32)
        xy_out = np.empty((T, 2), dtype=np.float32)
        yaw_out = np.empty(T, dtype=np.float32)
        di = 0
        for t, (is_d, i) in enumerate(rows):
            if is_d:
                scans[t] = scans_d[di]; di += 1
                xy_out[t] = dxy[i]; yaw_out[t] = dyaw_all[i]
            else:
                scans[t] = scans_f[i]
                xy_out[t] = fill_xy[i]; yaw_out[t] = fill_yaw[i]
        hold = int(getattr(self, '_nav_goal_hold', 0))
        if hold > 0:                                    # park at the last real pose
            scans = np.concatenate([scans, np.repeat(scans[-1:], hold, axis=0)], 0)
            xy_out = np.concatenate([xy_out, np.repeat(xy_out[-1:], hold, axis=0)], 0)
            yaw_out = np.concatenate([yaw_out, np.repeat(yaw_out[-1:], hold)], 0)
        sd = np.asarray(snap_ds) if snap_ds else np.asarray([np.nan])
        end_gap = float(np.linalg.norm(xy_out[-1].astype(np.float64) - rxy[n_r - 1]))
        utils.print_color(
            f'[gridnav/A2] ribbon: {T} frames ({len(d_idx)} real / {n_fill} synth-fill), '
            f'{n_seg} segs, snap p50={np.nanmedian(sd):.2f} p90={np.nanpercentile(sd, 90):.2f} wu, '
            f'{n_relax} yaw-relaxed, end-gap {end_gap:.2f} wu', c='g')
        return dict(scans=scans, xy=xy_out, yaw=yaw_out, cells=route['cells'])

    def _nav_build_plan_tfexec(self):
        """Executor diagnostic: real dataset scan sequence starting near the ant.

        Among the frames nearest the ant, prefer one with a LONG remaining
        in-episode future (>= nav_tf_min_frames), so the ribbon spans real
        distance -- the nearest frame alone often sits at an episode's tail
        (observed: 1-100-frame micro-segments, ~0.1-0.7 cell spans)."""
        data = self._load_tf_data()
        ep_id = data['ep_id']
        if 'ep_end' not in data:                     # last in-episode index per frame
            N = len(ep_id)
            ep_end = np.empty(N, dtype=np.int64)
            last = N - 1
            for i in range(N - 1, -1, -1):
                if i < N - 1 and ep_id[i] != ep_id[i + 1]:
                    last = i
                ep_end[i] = last
            data['ep_end'] = ep_end
        ep_end = data['ep_end']
        txy = np.asarray(self.env.get_xy(), dtype=np.float32)    # diagnostic: true xy
        d2 = ((data['xy'] - txy[None, :]) ** 2).sum(1)
        min_len = int(getattr(self.args, 'nav_tf_min_frames', 120))
        ## HEADING GATE (2026-07-05): also require the dataset ant's heading at the
        ## segment start to roughly match the live ant's (proprio quat, in-spec).
        ## Without it the ribbon's first goal scans can be rotated up to 180deg vs
        ## the live scan -- an obs->goal yaw delta the k=3-4-trained inv-dyn never
        ## saw. Relaxed (old behavior) if no candidate matches.
        max_dyaw = float(getattr(self.args, 'nav_tf_max_dyaw', 60.0)) * np.pi / 180.0
        cands = np.argsort(d2)[:2000]
        ant_yaw = self._nav_proprio_yaw()
        try:
            _, cyaw = obs_to_xy_yaw(data['obs'][cands], has_quat=True)
            dyaw = np.abs(np.arctan2(np.sin(cyaw - ant_yaw), np.cos(cyaw - ant_yaw)))
        except Exception:  # noqa: BLE001
            dyaw = np.zeros(len(cands))
        i0 = None
        for pass_dyaw in (max_dyaw, np.inf):          # strict pass, then relaxed
            for k, i in enumerate(cands):             # nearest first, need a long future
                i = int(i)
                if ep_end[i] - i >= min_len and dyaw[k] <= pass_dyaw:
                    i0 = i
                    break
            if i0 is not None:
                break
        if i0 is None:                                # fallback: plain nearest
            i0 = int(np.argmin(d2))
        i1 = int(ep_end[i0])
        seg = data['obs'][i0:i1 + 1]
        scans = self.scanner.scan_obs(seg, has_quat=True).astype(np.float32)
        ## pursuit bookkeeping
        self._nav_plan_xy = np.asarray(data['xy'][i0:i1 + 1], dtype=np.float64)
        self._nav_new_plan = True
        span = float(np.linalg.norm(data['xy'][i1] - data['xy'][i0]))
        utils.print_color(
            f'[gridnav/tf_exec] real segment: {len(seg)} frames, start '
            f'{np.sqrt(d2[i0]):.1f}mj from ant, span {span:.1f}mj '
            f'({span / 4.0:.1f} cells)', c='g')
        return scans

    def _nav_gen_cond_stgl(self, g_cond=None, b_s=None, **kw):
        """Wrapper installed over ``policy.gen_cond_stgl`` when gridnav is on.

        The bypass paths also append a timing entry to the policy's
        ``ncp_pred_time_list``: the end-of-run summary indexes that array 2-D and
        CRASHES on an empty list (observed: all 3 arms lost their jsons to
        ``IndexError`` in ``get_avg_sampling_time``)."""
        try:
            if self.nav_mode in ('bfs', 'tf_exec'):
                _t0 = time.time()
                _plan = (self._nav_build_plan_bfs() if self.nav_mode == 'bfs'
                         else self._nav_build_plan_tfexec())
                try:
                    self.policy.ncp_pred_time_list.append([0, time.time() - _t0])
                except Exception:  # noqa: BLE001
                    pass
                return SimpleNamespace(pick_traj=_plan)
            if self.nav_mode == 'subgoal' and self._nav_pending is not None:
                n_prev = self.policy.n_comp
                self.policy.n_comp = self._nav_pending['n_comp']
                try:
                    return self._nav_real_gen(g_cond=g_cond, b_s=b_s, **kw)
                finally:
                    self.policy.n_comp = n_prev
        except Exception as e:  # noqa: BLE001
            utils.print_color(f'[gridnav] plan build FAILED -> diffusion ({e})', c='y')
        return self._nav_real_gen(g_cond=g_cond, b_s=b_s, **kw)

    def _nav_subgoal_band(self):
        """Plan B: grid-routed SUBGOAL band ~N cells ahead along the BFS route.

        Returns (K, nb) band + stores the adapted n_comp, or None to fall back to
        the ordinary full-goal band."""
        if self._nav_goal is None:
            return None
        st = self._nav_est_pose()
        route = self._nav_route_from_est(st)
        if route is None:
            return None
        ## clip the route at the subgoal distance (frames are step_wu apart)
        n_sub = int(round(self._nav_subgoal_cells * self.scanner.maze_unit
                          / self._nav_synth.step_wu))
        end = min(max(n_sub, 2), len(route['xy']) - 1)
        sub_route = dict(xy=route['xy'][:end + 1], yaw=route['yaw'][:end + 1])
        band = self._nav_synth.band_along_route(
            sub_route, self.goal_band_k, self.goal_band_step)
        ## adapt composition to the clipped distance (~3.4 cells per segment)
        dist_cells = (end * self._nav_synth.step_wu) / self.scanner.maze_unit
        n_comp = int(np.clip(int(np.ceil(dist_cells / 3.4)) + 1, 2, self.policy.n_comp))
        self._nav_pending = dict(n_comp=n_comp)
        utils.print_color(
            f"[gridnav/subgoal] est=({st['est_xy'][0]:.1f},{st['est_xy'][1]:.1f}) "
            f'conf={st["conf"]:.3f} subgoal@{dist_cells:.1f} cells n_comp={n_comp}', c='g')
        return band

    def _goal_scan_band_route(self, gl_pos):
        """Eval goal band rendered along the BFS goal-approach corridor (the
        train/eval OOD fix): last-K strided scans of the *real route into the
        goal*, heading = route tangent. Falls back to the straight-line synth."""
        try:
            self._ensure_grid_loc()
            gxy = np.asarray(gl_pos, dtype=np.float64).reshape(-1)[:2]
            ## approach FROM the ant's side so the band covers the true entry corridor
            sxy = np.asarray(self.env.get_xy(), dtype=np.float64) \
                if not getattr(self, '_nav_live', None) or self._nav_live.est_xy is None \
                else np.asarray(self._nav_live.est_xy, dtype=np.float64)
            synth = getattr(self, '_nav_synth', None)
            if synth is None:
                from diffuser.datasets.lidar.lidar_route_nav import RouteScanSynthesizer
                synth = self._nav_synth_fallback = getattr(
                    self, '_nav_synth_fallback', None) or RouteScanSynthesizer(self.scanner)
            route = synth.synthesize(start_xy=sxy, goal_xy=gxy, goal_hold=0)
            if route is None or len(route['xy']) < 2:
                return None
            return synth.band_along_route(route, self.goal_band_k, self.goal_band_step)
        except Exception as e:  # noqa: BLE001
            utils.print_color(f'[band_from_route] failed -> synth band ({e})', c='y')
            return None

    # -- PM Stage-3: live LiDAR self-localization + goal-arrival ------------#
    def _setup_arrival(self):
        """Enable the PM's closed-loop rule: the ant decodes its position from LiDAR
        (histogram filter on the (x,y,theta) grid) and ENDS the run when it enters the
        goal cell (its scan matches the trained goal scan). LIDAR_ARRIVAL=1 turns it on;
        default off leaves the rollout unchanged. Reuses the rerank grid/localizer if
        present, else builds one."""
        self._arrival_on = bool(int(getattr(self.args, 'lidar_arrival', 0)))
        self._arrival_det = None
        self._arrival_ends = False
        if not self._arrival_on:
            return
        try:
            if getattr(self, 'loc', None) is None:
                self.loc_grid = load_or_build_grid(
                    self.env.name, self.lidar_config,
                    xy_step=float(getattr(self.args, 'loc_grid_xy_step', 1.0)),
                    n_theta=int(getattr(self.args, 'loc_grid_n_theta', 12)),
                    cache_root=getattr(self.args, 'loc_grid_cache', 'data/ogb_maze/lidar_grid'))
                self.loc = GridLocalizer(
                    self.loc_grid, sigma=float(getattr(self.args, 'loc_grid_sigma', 1.0)))
            from diffuser.datasets.lidar.lidar_arrival import LidarArrivalDetector
            self._ArrivalCls = LidarArrivalDetector
            self._arrival_ends = bool(int(getattr(self.args, 'lidar_arrival_ends', 1)))
            self._arrival_goal_mode = str(getattr(self.args, 'lidar_arrival_goal_mode', 'lidar')).lower()
            self._arrival_persist = int(getattr(self.args, 'lidar_arrival_persist', 3))
            self._arrival_motion_sigma = float(getattr(self.args, 'loc_motion_sigma', 1.0))
            utils.print_color(
                f'[arrival] ON: ends_run={self._arrival_ends} goal_mode={self._arrival_goal_mode} '
                f'persist={self._arrival_persist}', c='c')
        except Exception as e:  # noqa: BLE001
            self._arrival_on = False
            utils.print_color(f'[arrival] setup FAILED -> disabled ({e})', c='y')

    def _lidar_arrival_reset(self, gl_pos):
        """Per-episode: build the goal scan + goal cell and a fresh arrival detector."""
        if getattr(self, 'nav_mode', ''):
            self._nav_ep_reset(gl_pos)
        if not getattr(self, '_arrival_on', False):
            return
        try:
            gx, gy = float(gl_pos[0]), float(gl_pos[1])
            yaw = 0.0 if isinstance(self.goal_scan_yaw, str) else float(self.goal_scan_yaw)
            goal_scan = self.scanner.scan(gx, gy, yaw).astype(np.float32)
            if self._arrival_goal_mode == 'true':
                gi, gj = self.scanner.world_to_cell(gx, gy)
                goal_cell = (int(gi), int(gj))
            elif getattr(self, '_nav_goal', None) is not None:
                ## CONSISTENCY: stop at the SAME cell the navigator is driving to
                ## (observed: nav aimed at one decoy while arrival watched another).
                ## _nav_ep_reset ran just above, so _nav_goal is this episode's.
                goal_cell = tuple(int(v) for v in self._nav_goal['cell'])
            else:  # pure LiDAR: localize the goal scan on the grid
                goal_cell = tuple(int(v) for v in self.loc.localize_single(goal_scan)['cell'])
            self._arrival_det = self._ArrivalCls(
                self.loc, goal_scan, goal_cell,
                motion_sigma_cells=self._arrival_motion_sigma,
                cell_radius=0, scan_tol=None, persist=self._arrival_persist, min_steps=5)
            self._arrival_fired = False
        except Exception as e:  # noqa: BLE001
            self._arrival_det = None
            utils.print_color(f'[arrival] reset failed ep -> off this ep ({e})', c='y')

    def _lidar_arrival_step(self, obs):
        """Per-step: update the live localizer from the current scan; return its dict
        (with ``arrived``). The base rollout loop uses ``arrived`` to end the episode.

        With verify-and-hop on, a self-declared arrival is NOT trusted: the live scan
        is matched against the goal band's tail scan (pure LiDAR). A match accepts
        (the run may end); a mismatch REJECTS the cell and re-routes to the next-best
        goal-posterior hypothesis, so the ant sequentially tests the decoy family
        instead of parking on hypothesis #1 (memory 2026-07-08 fix-1)."""
        if getattr(self, 'nav_mode', ''):
            self._nav_step_track(obs)
        if not getattr(self, '_arrival_on', False) or self._arrival_det is None:
            return None
        try:
            scan = self._obs_to_scan(obs)                 # LiDAR at the real pose
            r = self._arrival_det.update(scan)
            if r.get('arrived') and not getattr(self, '_arrival_fired', False):
                if getattr(self, '_nav_verify_hop', False):
                    r = self._verify_arrival(scan, r)
                else:
                    self._arrival_fired = True
                    utils.print_color(
                        f"[arrival] LiDAR: entered goal cell {self._arrival_det.goal_cell} "
                        f"(est {r['est_cell']}, {self._arrival_det.live.n} steps)", c='g')
            return r
        except Exception as e:  # noqa: BLE001
            utils.print_color(f'[arrival] step err ({e})', c='y')
            return None

    def _verify_arrival(self, scan, r):
        """Accept or reject a self-declared arrival (verify-and-hop, fix-1).

        Compares the LIVE scan at the arrived cell to the goal band's tail scan
        (the fixed, pure-LiDAR goal descriptor -- unchanged across hops; only the
        ROUTING target moves). Match (<= verify_tol per-beam RMS wu) -> accept and
        let the run end. Mismatch -> reject this cell, exclude it from the posterior,
        and hop to the next-best hypothesis. Mutates and returns ``r`` (``arrived``
        set False to suppress the end while hopping)."""
        g = getattr(self, '_nav_goal', None)
        tail = None if g is None else g.get('band_tail', None)
        cur_cell = tuple(int(v) for v in self._arrival_det.goal_cell)
        ## belief-converged gate (fix-2): don't spend a hop on an unconverged phantom
        ## (e.g. the step-61 spawn family). Suppress until the posterior mass is high.
        if (self._nav_verify_min_conf > 0.0
                and float(r.get('conf', 1.0)) < self._nav_verify_min_conf):
            utils.print_color(
                f"[verify-hop] arrival at {cur_cell} but conf {r.get('conf', 0.0):.3f} "
                f"< {self._nav_verify_min_conf} -> wait (belief not converged)", c='y')
            r['arrived'] = False
            self._arrival_det._hit_run = 0                # require a fresh persist run
            return r
        if tail is None:                                  # no descriptor -> can't verify
            self._arrival_fired = True
            return r
        gd = float(np.sqrt(np.mean(
            (np.asarray(scan, dtype=np.float32) - np.asarray(tail, dtype=np.float32)) ** 2)))
        if gd <= self._nav_verify_tol:                    # ACCEPT
            self._arrival_fired = True
            utils.print_color(
                f"[verify-hop] ACCEPT at {cur_cell}: live-vs-goal scan {gd:.3f} "
                f"<= tol {self._nav_verify_tol} (true={g.get('true_cell')}, "
                f"hops_used={self._nav_hop_used})", c='g')
            return r
        ## REJECT -> exclude this cell + hop to the next posterior hypothesis
        self._nav_rejected_cells.add(cur_cell)
        nxt = self._nav_hop_to_next()
        if nxt is None:                                   # swept the family / budget spent
            self._arrival_fired = True
            utils.print_color(
                f"[verify-hop] REJECT at {cur_cell} (scan {gd:.3f} > tol "
                f"{self._nav_verify_tol}); no hypothesis left / budget spent "
                f"({self._nav_hop_used}/{self._nav_hop_budget}) -> accept as-is", c='y')
            return r
        utils.print_color(
            f"[verify-hop] REJECT {cur_cell} (scan {gd:.3f} > tol {self._nav_verify_tol}) "
            f"-> hop {self._nav_hop_used}/{self._nav_hop_budget} to {nxt['cell']} "
            f"(post {nxt['score']:.3f})", c='c')
        r['arrived'] = False                              # suppress the end; keep hopping
        return r

    def _nav_hop_to_next(self):
        """Re-point the navigator + arrival detector at the next-best un-rejected
        goal-posterior cell. Keeps the fixed goal descriptor (``true_cell`` /
        ``band_tail``) so verification still tests against the REAL goal. Returns the
        chosen hypothesis dict, or None if the budget is spent or none remain."""
        g = getattr(self, '_nav_goal', None)
        if g is None or self._nav_hop_used >= self._nav_hop_budget:
            return None
        nxt = next((h for h in g.get('hyps', [])
                    if tuple(int(v) for v in h['cell']) not in self._nav_rejected_cells),
                   None)
        if nxt is None:
            return None
        self._nav_hop_used += 1
        ## move only the ROUTING target (cell + endpoint); descriptor stays fixed
        g['cell'] = tuple(int(v) for v in nxt['cell'])
        g['xy'] = (float(nxt['xy'][0]), float(nxt['xy'][1]))
        ## re-arm the arrival detector on the new cell (fresh belief for the new leg)
        if getattr(self, '_arrival_det', None) is not None:
            try:
                yaw = (0.0 if isinstance(self.goal_scan_yaw, str)
                       else float(self.goal_scan_yaw))
                new_scan = self.scanner.scan(g['xy'][0], g['xy'][1], yaw).astype(np.float32)
                self._arrival_det = self._ArrivalCls(
                    self.loc, new_scan, g['cell'],
                    motion_sigma_cells=self._arrival_motion_sigma,
                    cell_radius=0, scan_tol=None,
                    persist=self._arrival_persist, min_steps=5)
            except Exception as e:  # noqa: BLE001
                utils.print_color(f'[verify-hop] re-arm arrival failed ({e})', c='y')
        self._arrival_fired = False
        ## force a prompt re-route to the new hypothesis (next replan tick)
        self._nav_first_plan = True
        self._nav_pending = None
        self._nav_cursor = 0
        self._nav_force_replan = True
        return nxt

    # -- normalizers -------------------------------------------------------#
    def _setup_normalizers(self):
        """Load the real LiDAR dataset normalizers instead of the hard-coded
        (x, y) ones.

        ``full_normalizer`` is the inverse-dynamics dataset normalizer: 34-D
        ``[x, y, cos, sin, lidar]`` observations + 8-D actions. It is rebuilt
        from the inv-dyn run's saved ``dataset_config.pkl`` so it matches
        training exactly (the expensive LiDAR ray-casting is disk-cached).

        ``train_normalizer`` (the planner's 30-D LiDAR normalizer, also the
        normalizer the policy uses) is the LiDAR slice ``[4:]`` of the inv-dyn
        observation normalizer. Because both datasets are built from the *same*
        transitions with the *same* LiDAR scans (see ogb_sequence_dataset), this
        slice is bit-for-bit the planner's own LiDAR normalizer -- which also
        guarantees a generated plan waypoint is a valid inv-dyn goal.
        """
        inv_path = getattr(self.args, 'inv_model_path', None)
        assert inv_path is not None, (
            'LiDAR rollout needs args.inv_model_path set to the LiDAR inv-dyn run '
            'dir (see the dfu_ndim==30 / lidar branch in plan_ogb_stgl_sml.py).')

        utils.print_color(
            f'[lidar planner] building inv-dyn normalizer from dataset of {inv_path}', c='c')
        inv_cfg = load_config(inv_path, 'dataset_config.pkl')
        ## n_beams (== the LiDAR-slice width, always the trailing dims of the obs).
        ## Read from the inv-dyn dataset config so this works for any obs_layout
        ## (xyo_lidar -> offset 4, full_lidar -> offset 29).
        try:
            self.n_beams = int(inv_cfg._dict['dataset_config']['lidar_config']['n_beams'])
        except Exception:  # noqa: BLE001
            self.n_beams = int(getattr(getattr(self, 'diffusion', None), 'observation_dim', 30))
        inv_dataset = inv_cfg()
        self.full_normalizer = inv_dataset.normalizer

        obs_dim = self.full_normalizer.normalizers['observations'].mins.shape[0]
        self.lidar_offset = obs_dim - self.n_beams        ## leading non-LiDAR width
        lidar_idxs = tuple(range(self.lidar_offset, obs_dim))  ## the trailing LiDAR slice
        self.train_normalizer = self._slice_obs_normalizer(self.full_normalizer, lidar_idxs)

        ## free the (large) dataset; we only needed its normalizer
        try:
            del inv_dataset
        except Exception:  # noqa: BLE001
            pass

        self.check_is_same_nmlizer()

    def _slice_obs_normalizer(self, full_norm, obs_idxs):
        """Return a DatasetNormalizer whose 'observations' normalizer is the
        ``obs_idxs`` slice of ``full_norm``'s observation normalizer (actions
        carried over unchanged). For LimitsNormalizer this exactly reproduces a
        normalizer fit on that observation slice."""
        obs_n = full_norm.normalizers['observations']
        act_n = full_norm.normalizers['actions']
        obs_idxs = list(obs_idxs)
        data_dict = {
            'observations': np.stack([obs_n.mins[obs_idxs], obs_n.maxs[obs_idxs]]).astype(np.float32),
            'actions': np.stack([act_n.mins, act_n.maxs]).astype(np.float32),
        }
        ## eval_solo=True -> build straight from the [min; max] rows (no path flatten)
        return DatasetNormalizer(data_dict, type(obs_n).__name__, eval_solo=True)

    def check_is_same_nmlizer(self):
        """The planner's n_beams-D LiDAR normalizer must equal the trailing LiDAR
        slice ``[lidar_offset:]`` of the inv-dyn normalizer (true by construction).
        lidar_offset is 4 for xyo_lidar and 29 for full_lidar."""
        tr = self.train_normalizer.normalizers['observations']
        fl = self.full_normalizer.normalizers['observations']
        n = tr.mins.shape[0]
        off = self.lidar_offset
        assert np.allclose(tr.mins, fl.mins[off:off + n]) and \
               np.allclose(tr.maxs, fl.maxs[off:off + n]), \
            'planner LiDAR normalizer != inv-dyn LiDAR slice'

    # -- live obs -> LiDAR helpers ----------------------------------------#
    def _obs_xy_yaw(self, obs):
        xy, yaw = obs_to_xy_yaw(obs[None], has_quat=True)
        return float(xy[0, 0]), float(xy[0, 1]), float(yaw[0])

    def _obs_to_scan(self, obs):
        """live 29-D ant obs -> (30,) LiDAR scan at the real heading."""
        x, y, yaw = self._obs_xy_yaw(obs)
        return self.scanner.scan(x, y, yaw).astype(np.float32)

    def _obs_to_invdyn_obs(self, obs):
        """live raw ant obs -> the inv-dyn composite obs, matching the TRAINED layout.

        full_lidar -> [full raw obs (incl. joints + velocities), lidar]  (proprioception)
        xyo_lidar  -> [x, y, cos(yaw), sin(yaw), lidar]                   (car-like, legacy)

        Which layout to build is decided by ``self.lidar_offset`` -- the inv-dyn obs'
        non-LiDAR prefix width, read authoritatively from the inv-dyn normalizer in
        ``_setup_normalizers`` (4 for xyo_lidar, 29 for full_lidar). NOTE: do NOT key
        off ``self.lidar_config['obs_layout']`` -- that is the *planner's* layout
        ('lidar', a pure 30-D scan), not the inv-dyn's, and using it silently built the
        legacy 34-D obs against the 59-D full_lidar normalizer (shape (34,) vs (59,)).
        """
        lidar = self._obs_to_scan(obs)                          # scan at the real pose
        if self.lidar_offset == N_XYO:                          # 4 -> legacy xyo_lidar
            x, y, yaw = self._obs_xy_yaw(obs)
            head = np.array([x, y, np.cos(yaw), np.sin(yaw)], dtype=np.float32)
        else:                                                   # full_lidar (proprioception)
            head = np.asarray(obs, dtype=np.float32)[:self.lidar_offset]
        return np.concatenate([head, lidar]).astype(np.float32)

    def _goal_scan(self, gl_pos):
        """goal *position* -> (30,) goal LiDAR scan at the chosen yaw."""
        gx, gy = float(gl_pos[0]), float(gl_pos[1])
        yaw_mode = self.goal_scan_yaw
        if isinstance(yaw_mode, str):
            if yaw_mode == 'goal':
                ## gl_pos is a full obs -> use the goal state's real heading
                _, yaw = obs_to_xy_yaw(np.asarray(gl_pos)[None], has_quat=True)
                yaw = float(yaw[0])
            else:
                raise NotImplementedError(f'lidar_goal_yaw={yaw_mode!r}')
        else:
            yaw = float(yaw_mode)
        ## the yaw used to RENDER this scan is known to the code even though the
        ## model never sees it -- stash it so localization can constrain candidate
        ## cells to the SAME orientation instead of searching all of them (2026-07-30).
        self._nav_goal_band_yaw = float(yaw)
        return self.scanner.scan(gx, gy, yaw).astype(np.float32)

    def _goal_scan_band(self, gl_pos):
        """goal position -> (K, n_beams) LiDAR 'band': K scans stepping INTO the
        goal along the corridor the goal scan shows as most open. K=1 -> (1,
        n_beams), identical to the single goal scan. PURE LiDAR: only scans are
        produced; the goal (x, y) is used solely to synthesize them, exactly as
        the single-scan path already does (the model never sees x, y)."""
        K = self.goal_band_k
        if getattr(self, 'tf_band', False) and K > 1:
            return self._goal_scan_band_teacher(gl_pos)
        ## band_from_route: render the band along the BFS goal-approach corridor
        ## (curved route, heading = route tangent) instead of the straight-line
        ## most-open-beam walk -- matches the strided-band training distribution.
        if getattr(self, 'band_from_route_on', False) and K > 1:
            _band = self._goal_scan_band_route(gl_pos)
            if _band is not None:
                return _band
        goal_scan = self._goal_scan(gl_pos)              ## (n_beams,) at the goal pose
        if K <= 1:
            return goal_scan[None].astype(np.float32)
        gx, gy = float(gl_pos[0]), float(gl_pos[1])
        ## resolve the goal-scan heading the same way _goal_scan does
        if isinstance(self.goal_scan_yaw, str):
            _, _yaw = obs_to_xy_yaw(np.asarray(gl_pos)[None], has_quat=True)
            yaw = float(_yaw[0])
        else:
            yaw = float(self.goal_scan_yaw)
        ## most-open beam == the corridor; step back along it to synthesize the approach
        b = int(np.argmax(goal_scan))
        ang = yaw + float(self.scanner.beam_offsets[b])
        dx, dy = np.cos(ang), np.sin(ang)
        step = float(self.goal_band_step)
        open_len = float(goal_scan[b])
        band = []
        for j in range(K):                               ## earliest (H-K) -> goal (H-1)
            back = min((K - 1 - j) * step, max(open_len - step, 0.0))
            band.append(self.scanner.scan(gx - back * dx, gy - back * dy, yaw))
        return np.asarray(band, dtype=np.float32)         ## (K, n_beams)

    # -- teacher-forced real goal band (diagnostic) ------------------------#
    def _load_tf_data(self):
        """Lazy-load a real full-state dataset; cache per-frame xy + episode id.
        Used only when ``tf_band`` is set. Scans are re-rendered with the SAME
        scanner that built the training data, so the band is the real last-K
        training scans of an approach into the goal cell (no x,y reaches the model)."""
        if self._tf_cache is not None:
            return self._tf_cache
        path = self.tf_npz
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'[tf_band] dataset npz not found: {path!r}. Set LIDAR_TF_NPZ to a '
                f'real {self.env.name} dataset with full-state "observations".')
        d = np.load(path, allow_pickle=True)
        if 'observations' not in d.files:
            raise KeyError(f'[tf_band] {path} has no "observations" key; found {list(d.files)}')
        obs = np.asarray(d['observations'], dtype=np.float32)
        term = None
        for k in ('terminals', 'dones', 'timeouts'):
            if k in d.files:
                term = np.asarray(d[k]).reshape(-1).astype(bool)
                break
        if term is not None and term.shape[0] == obs.shape[0]:
            ep_id = np.cumsum(np.concatenate([[0], term[:-1].astype(np.int64)]))
        else:
            ep_id = np.zeros(obs.shape[0], dtype=np.int64)
            utils.print_color('[tf_band] no terminals in npz -> treating as ONE episode '
                              '(band may rarely cross a real episode start)', c='y')
        self._tf_cache = dict(obs=obs, xy=obs[:, :2].astype(np.float32), ep_id=ep_id)
        utils.print_color(f'[tf_band] loaded {path}: N={obs.shape[0]} '
                          f'obs_dim={obs.shape[1]} episodes={int(ep_id.max()) + 1}', c='c')
        return self._tf_cache

    def _goal_scan_band_teacher(self, gl_pos):
        """goal position -> (K, n_beams) band = the TRUE last-K scans of a real
        trajectory ending nearest the goal (same episode), rendered with self.scanner.

        ``tf_band_stride`` (S) spaces the K band scans S dataset-frames apart so the
        band spans SEVERAL cells of the real approach (wide-baseline goal context)
        instead of the last K *consecutive* frames (<1 cell). S=1 == original behaviour."""
        K = self.goal_band_k
        S = max(int(getattr(self, 'tf_band_stride', 1)), 1)
        need = (K - 1) * S                               # earliest predecessor offset
        data = self._load_tf_data()
        gxy = np.asarray(gl_pos, dtype=np.float32)[:2]
        d2 = ((data['xy'] - gxy[None, :]) ** 2).sum(1)
        ep_id = data['ep_id']
        chosen = None
        for i in np.argsort(d2)[:5000]:                  # nearest frames to the goal
            i = int(i)
            if i - need >= 0 and ep_id[i - need] == ep_id[i]:
                chosen = i
                break
        if chosen is None:                               # fallback: clamp at the front
            chosen = int(np.argmin(d2))
            utils.print_color('[tf_band] no K strided in-episode predecessors near goal; '
                              'clamping front (band repeats the first frame)', c='y')
        ## anchor the goal-nearest frame at `chosen`, step back S frames (K-1) times,
        ## drop any that fall before the episode start, order earliest -> goal.
        idx = [chosen - j * S for j in range(K)]
        idx = [k for k in idx if k >= 0 and ep_id[k] == ep_id[chosen]][::-1]
        seg = data['obs'][np.asarray(idx, dtype=int)]    # (<=K, obs_dim) earliest->goal
        if seg.shape[0] < K:                             # pad at the front if clamped
            seg = np.concatenate([np.repeat(seg[:1], K - seg.shape[0], axis=0), seg], axis=0)
        band = self.scanner.scan_obs(seg, has_quat=True).astype(np.float32)   # (K, n_beams)
        ## the REAL recorded heading of the goal-anchor frame -- known to the code
        ## (it's how `band`'s last scan was rendered) even though it's never fed to
        ## the model. Stashed so localization can lock candidate cells to this same
        ## orientation instead of letting each one pick its own best-fitting theta,
        ## which is exactly what a 90deg-rotated look-alike room exploits (2026-07-30).
        try:
            _, _band_yaws = obs_to_xy_yaw(seg, has_quat=True)
            self._nav_goal_band_yaw = float(_band_yaws[-1])
        except Exception:  # noqa: BLE001
            self._nav_goal_band_yaw = None
        dmj = float(np.sqrt(d2[chosen]))
        span = float(np.linalg.norm(data['xy'][idx[-1]] - data['xy'][idx[0]]))
        utils.print_color(f'[tf_band] goal=({gxy[0]:.2f},{gxy[1]:.2f}) -> frame {chosen} '
                          f'@ {dmj:.2f} mj ({dmj / 4.0:.2f} cells); stride={S} '
                          f'band_span={span:.2f} mj ({span / 4.0:.2f} cells); band {band.shape}', c='c')
        return band

    # -- overridden hooks --------------------------------------------------#
    def _get_planner_stgl_input(self, obs_cur, gl_pos):
        ## condition the diffusion planner on the start scan and the goal *band*
        ## (K scans into the goal; K=1 == the original single goal scan).
        ## gridnav: cache the plan-time obs/goal for the wrapped generator, and in
        ## 'subgoal' mode condition on a grid-routed subgoal band instead.
        self._nav_obs_cur, self._nav_gl_pos = obs_cur, gl_pos
        if getattr(self, 'nav_mode', '') == 'subgoal':
            try:
                band = self._nav_subgoal_band()
                if band is not None:
                    return self._obs_to_scan(obs_cur), band
            except Exception as e:  # noqa: BLE001
                utils.print_color(f'[gridnav/subgoal] band FAILED -> goal band ({e})', c='y')
            self._nav_pending = None
        if getattr(self, '_loc_rerank_on', False):
            ## cache the BFS graph-distance-to-goal field for the plan re-ranker
            ## (goal cell from the eval goal position -- eval-side bookkeeping only;
            ## the diffusion model still sees only the LiDAR goal band).
            gi, gj = self.scanner.world_to_cell(
                np.asarray([gl_pos[0]]), np.asarray([gl_pos[1]]))
            goal_cell = (int(gi[0]), int(gj[0]))
            self._rerank_ctx = dict(goal_cell=goal_cell,
                                    dist_field=self._bfs_dist_field(goal_cell))
        return self._obs_to_scan(obs_cur), self._goal_scan_band(gl_pos)

    def _get_invdyn_obs_nm(self, obs_cur):
        ## the inv-dyn consumes the composite obs (59-D full_lidar / 34-D legacy xyo),
        ## normalized by full_normalizer
        invdyn_obs = self._obs_to_invdyn_obs(obs_cur)
        return self.full_normalizer.normalize(invdyn_obs, 'observations')

    def _get_subgoal_log_dist(self, obs_cur, goal_cur):
        ## LiDAR-space distance between the current scan and the subgoal scan
        ## (the (x, y) slice used by the base would be two beam distances here)
        return float(np.linalg.norm(self._obs_to_scan(obs_cur) - goal_cur))