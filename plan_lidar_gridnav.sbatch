#!/bin/bash
# --- SLURM Settings ---
#SBATCH --job-name=lidar_gridnav
#SBATCH --output=slurm_logs/lidar_gridnav-%j.out
#SBATCH --error=slurm_logs/lidar_gridnav-%j.out
#SBATCH --partition=part-preempt
#SBATCH --qos=qos-preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --exclude=ece-ykasten1

# ============================================================================
# OP 2 -- GRID NAVIGATOR (Plan A): the PM's likelihood grid CLOSES THE LOOP.
#
# Every replan: the ant localizes ITSELF from its live scans (histogram filter on
# the (x,y,theta) grid), localizes the GOAL from the goal scan band, BFS-routes over
# free cells, renders the scan sequence along that route, and the existing
# waypoint-follower + inv-dyn execute it. LIDAR_ARRIVAL ends the run when the
# localized cell enters the goal cell. Navigation is scans-only end to end; (x,y) is
# used solely to render scans (the repo standard set by the goal scan + grid).
#
# LIDAR_NAV_MODE=bfs BYPASSES diffusion sampling (the plan IS the routed scan
# sequence), so the checkpoint quality does not matter here -- this isolates
# localization + routing + execution. Three arms:
#   (0) tfexec     - executor sanity (5 ep): plan = a REAL dataset scan sequence
#                    starting near the ant. If the ant can't follow real scans,
#                    fix execution before anything else.
#   (1) nav_true   - Plan A with the TRUE goal cell (diagnostic: isolates
#                    self-localization + routing + execution from goal ambiguity).
#   (2) nav_lidar  - Plan A fully pure-LiDAR (goal cell from the goal scan band)
#                    + LIDAR_ARRIVAL=1 (the ant stops itself). The headline arm.
#
# Read ep_srate / ep_min_goal_dist. Expected: (1) produces the FIRST successes;
# (2) trails it only by goal-localization misses (logged per episode as
# '[gridnav] ep goal ... N cells off').
#
# SYNC FIRST: diffuser/datasets/lidar/lidar_route_nav.py (NEW),
# ogb_stgl_sml_lidar_planner_v1.py, plan_ogb_stgl_sml.py. Bytecode cleared below.
# RUN IN PARALLEL with plan_lidar_tfband_gate.sbatch (separate GPU) -- independent.
# ============================================================================

mkdir -p slurm_logs rollout_results
source ~/.bashrc
source /home/projects/crml-prj10233/miniforge3/etc/profile.d/conda.sh
conda activate compdfu_ogb_release
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
echo "Job started on $(hostname)"

## bfs mode does not sample the model, so ANY existing checkpoint dir works;
## default = the original (K=1) run that is guaranteed to exist on the server.
CONFIG=${CONFIG:-config/ogb_ant_maze/og_antM_Gi_lidar30_Cd_Stgl_PadBuf_Ft64_ts512.py}
PLANS_DIR="logs/antmaze-giant-stitch-v0/plans"

find . -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null
find . -name '*.pyc' -delete 2>/dev/null

# ---- fail fast: new module must import; grid cache pre-built (CPU, ~1s) ----
python -c "from diffuser.datasets.lidar.lidar_route_nav import RouteScanSynthesizer, bfs_cell_path; \
from diffuser.datasets.lidar.lidar_grid import load_or_build_grid; \
load_or_build_grid('antmaze-giant-stitch-v0', dict(n_beams=30, max_range=12.0), xy_step=1.0, n_theta=12)" \
  || { echo '!! route-nav import / grid build failed'; exit 1; }

# ---- shared knobs ----
export LIDAR_REPLAN=1
export LIDAR_REPL_EVERY=${LIDAR_REPL_EVERY:-250}   # re-aim every 250 steps (cheap: no sampling)
export LIDAR_MAX_N_REPL=15
export LIDAR_N_MAX_STEPS=2500
export LIDAR_GOAL_YAW=0.0
export LIDAR_LOC_XYSTEP=1.0
export LIDAR_LOC_NTHETA=12
export LIDAR_LOC_MOTION_SIGMA=1.0
export LIDAR_NAV_STEP_WU=${LIDAR_NAV_STEP_WU:-0.13}
export LIDAR_PICK_TYPE=first                       # bfs mode never reaches the picker

N_EP=${N_EP:-20}

run_plan () {           # $1 = tag, $2 = n_ep
    local TAG=$1; local NEP=${2:-$N_EP}
    local STAMP; STAMP=$(mktemp); touch "$STAMP"
    echo "=== [$TAG] NAV_MODE=$LIDAR_NAV_MODE GOAL=$LIDAR_NAV_GOAL_MODE ARRIVAL=${LIDAR_ARRIVAL:-0}, $NEP ep ==="
    PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=offline \
    python diffuser/ogb_task/ogb_maze_v1/plan_ogb_stgl_sml.py \
        --config "$CONFIG" --plan_n_ep "$NEP" --pl_seeds 0
    local rc=$?
    ## filter by this job's config tag so a PARALLEL job (other config) can't be picked up
    local CFG_TAG; CFG_TAG=$(basename "$CONFIG" .py)
    local SRC; SRC=$(find "$PLANS_DIR" -path "*${CFG_TAG}*" -name 00_rollout.json -newer "$STAMP" 2>/dev/null | head -1)
    rm -f "$STAMP"
    if [ $rc -ne 0 ] || [ -z "$SRC" ]; then echo "!! [$TAG] FAILED (exit $rc, json='$SRC')"; return 1; fi
    cp "$SRC" "rollout_results/00_rollout_${TAG}.json"
    echo "saved -> rollout_results/00_rollout_${TAG}.json"
    return 0
}

peek () {
python - "$1" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); g = d.get
md = g('ep_min_goal_dist'); suc = g('ep_is_suc')
print(f"[peek] {sys.argv[1]}  ep_srate={g('ep_srate')}  epoch={g('epoch_diffusion')}")
if isinstance(suc, list): print(f"   successes: {sum(bool(x) for x in suc)}/{len(suc)}")
if isinstance(md, list) and md:
    print(f"   ep_min_goal_dist: best={min(md):.1f}  mean={sum(md)/len(md):.1f}  worst={max(md):.1f} mj (1 cell~4mj)")
PY
}

# ---------------------------------------------------------------------------
# 2026-07-05 REGRESSION FIX: run 21129 was 0/65 with NO episode leaving the
# start corner -- pursuit v1 raced the cursor to the plan tail whenever the
# belief sat in a look-alike basin (systematic at ep start). Pursuit v2 is
# paced + snap-gated (see _nav_pick_wp_idx). Each rung now runs a PURSUIT=0
# control (pure time pacing = the regime that navigated 8 cells in 21089)
# next to the v2 arm; compare the pairs to confirm the fix and the pacing.
# Watch the new '[pursuit] t=.. cursor=..' lines: cursor should track ~t and
# snap=True should appear once the belief converges.
# ---------------------------------------------------------------------------

# (0a) EXECUTOR SANITY, PURSUIT OFF: follow a REAL scan sequence, time-paced.
#      THE decisive rung: if the ant can't traverse real ribbons here either,
#      the problem is execution (inv-dyn / yaw handoff), not pursuit.
export LIDAR_NAV_MODE=tf_exec
export LIDAR_NAV_GOAL_MODE=true
export LIDAR_ARRIVAL=0
export LIDAR_NAV_PURSUIT=0
if run_plan tfexec_p0 5; then peek rollout_results/00_rollout_tfexec_p0.json; fi

# (0b) EXECUTOR SANITY, PURSUIT v2 (paced + snap-gated correction).
export LIDAR_NAV_PURSUIT=1
if run_plan tfexec 5; then peek rollout_results/00_rollout_tfexec.json; fi

# (1a) PLAN A, TRUE GOAL CELL, PURSUIT OFF (localization+routing+execution).
export LIDAR_NAV_MODE=bfs
export LIDAR_NAV_GOAL_MODE=true
export LIDAR_ARRIVAL=0
export LIDAR_NAV_PURSUIT=0
if run_plan nav_true_p0; then peek rollout_results/00_rollout_nav_true_p0.json; fi

# (1b) PLAN A, TRUE GOAL CELL, PURSUIT v2. First successes expected in (1a)/(1b).
export LIDAR_NAV_PURSUIT=1
if run_plan nav_true; then peek rollout_results/00_rollout_nav_true.json; fi

# (2) PLAN A, PURE LiDAR + self-declared arrival (the PM loop, headline arm).
#     KNOWN (run 21089): the SYNTHESIZED goal band localizes 8-11 cells off on
#     look-alikes in 20/20 eps -> this arm navigates to the wrong cell. Kept as
#     the strict-pure baseline; arm (3) fixes the goal descriptor.
export LIDAR_NAV_MODE=bfs
export LIDAR_NAV_GOAL_MODE=lidar
export LIDAR_ARRIVAL=1
export LIDAR_ARRIVAL_ENDS=1
export LIDAR_ARRIVAL_GOAL_MODE=lidar
if run_plan nav_lidar; then peek rollout_results/00_rollout_nav_lidar.json; fi

# (3) PLAN A, pure LiDAR with a REAL goal band (LIDAR_TF_BAND=1): the goal is
#     described by the true last-K strided scans of a real approach (the natural
#     "this is what arriving looks like" input). The motion-history filter should
#     localize this band to the RIGHT cell -- watch '[gridnav] ep goal ... cells off'
#     drop from 8-11 to ~0-1. Still no (x,y) to the model/executor.
export LIDAR_TF_BAND=1
export LIDAR_TF_BAND_STRIDE=16
if run_plan nav_lidar_tf; then peek rollout_results/00_rollout_nav_lidar_tf.json; fi
export LIDAR_TF_BAND=0

echo "=== DONE. Ladder: (0a/0b) executor p0/v2 -> (1a/1b) nav_true p0/v2 -> (2) nav_lidar -> (3) nav_lidar_tf ==="
echo "  (0a) moves, (0b) moves -> executor healthy, pursuit v2 safe; use v2."
echo "  (0a) moves, (0b) stuck -> pursuit v2 still wrong; run everything PURSUIT=0."
echo "  (0a) stuck too         -> execution itself (inv-dyn OOD / yaw handoff):"
echo "      check x-range in the t-prints; try LIDAR_NAV_STEP_WU=0.10,"
echo "      LIDAR_NAV_LOOKAHEAD=6; then suspect ribbon-scan realism."
echo "  (1) >0 successes -> grid navigator works; gap of (2)/(3) vs (1) = goal-"
echo "      localization misses (see '[gridnav] ep goal ... cells off' lines)."
echo "  Any success -> scale up with plan_lidar_gridnav_bench100.sbatch."
