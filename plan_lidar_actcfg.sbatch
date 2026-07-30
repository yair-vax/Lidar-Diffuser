#!/bin/bash
# --- SLURM Settings ---
#SBATCH --job-name=lidar_actcfg
#SBATCH --output=slurm_logs/lidar_actcfg-%j.out
#SBATCH --error=slurm_logs/lidar_actcfg-%j.out
#SBATCH --partition=part-preempt
#SBATCH --qos=qos-preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --exclude=ece-ykasten1

# ============================================================================
# GOAL-CONTRAST GUIDANCE A/B (2026-07-05) -- the last eval-only lever.
#
# The sensitivity probe (invdyn_goal_sensitivity.json) measured the inv-dyn's
# goal channel at ~20% of action variance (stop-vs-go 0.11, 24-deg turn 0.16,
# 84-deg turn ignored) -> too weak to beat the gait autopilot, hence the six
# statistically-identical gridnav arms. LIDAR_ACT_CFG_W amplifies the channel
# at the action level:  a = a(s, g_self) + w * (a(s, g_subgoal) - a(s, g_self)).
#
# 2x2: w in {3, 6} x goal in {true, lidar-decoys}. THE READ IS SEPARATION:
# if nav_true_wN beats nav_lidar_wN (rank-sum on ep_min_goal_dist), plan content
# finally matters -> the channel is usable; tune w, add eps/seeds, then still
# consider the wide-k retrain for margin. If all four still sit at mean ~48 mj
# with no separation -> eval levers are DONE; launch the wide-k inv-dyn retrain
# (config og_antM_Gi_lidar30_full_g30d_invdyn_h64_widek.py, ready).
#
# SYNC FIRST: ogb_stgl_sml_planner_v1.py, ogb_stgl_sml_lidar_planner_v1.py,
# plan_ogb_stgl_sml.py (+ this file). Bytecode cleared below.
# ============================================================================

mkdir -p slurm_logs rollout_results
source ~/.bashrc
source /home/projects/crml-prj10233/miniforge3/etc/profile.d/conda.sh
conda activate compdfu_ogb_release
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
echo "Job started on $(hostname)"

CONFIG=${CONFIG:-config/ogb_ant_maze/og_antM_Gi_lidar30_Cd_Stgl_PadBuf_Ft64_ts512.py}
PLANS_DIR="logs/antmaze-giant-stitch-v0/plans"

find . -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null
find . -name '*.pyc' -delete 2>/dev/null

# ---- shared knobs (same as the gridnav ladder; pursuit v2 on) ----
export LIDAR_REPLAN=1
export LIDAR_REPL_EVERY=${LIDAR_REPL_EVERY:-250}
export LIDAR_MAX_N_REPL=15
export LIDAR_N_MAX_STEPS=2500
export LIDAR_GOAL_YAW=0.0
export LIDAR_LOC_XYSTEP=1.0
export LIDAR_LOC_NTHETA=12
export LIDAR_LOC_MOTION_SIGMA=1.0
export LIDAR_NAV_STEP_WU=${LIDAR_NAV_STEP_WU:-0.13}
export LIDAR_PICK_TYPE=first
export LIDAR_NAV_MODE=bfs
export LIDAR_ARRIVAL=0
export LIDAR_NAV_PURSUIT=1

N_EP=${N_EP:-20}

run_plan () {           # $1 = tag, $2 = n_ep
    local TAG=$1; local NEP=${2:-$N_EP}
    local STAMP; STAMP=$(mktemp); touch "$STAMP"
    echo "=== [$TAG] GOAL=$LIDAR_NAV_GOAL_MODE ACT_CFG_W=$LIDAR_ACT_CFG_W, $NEP ep ==="
    PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=offline \
    python diffuser/ogb_task/ogb_maze_v1/plan_ogb_stgl_sml.py \
        --config "$CONFIG" --plan_n_ep "$NEP" --pl_seeds 0
    local rc=$?
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
print(f"[peek] {sys.argv[1]}  ep_srate={g('ep_srate')}")
if isinstance(suc, list): print(f"   successes: {sum(bool(x) for x in suc)}/{len(suc)}")
if isinstance(md, list) and md:
    print(f"   ep_min_goal_dist: best={min(md):.1f}  mean={sum(md)/len(md):.1f}  worst={max(md):.1f} mj (1 cell~4mj)")
PY
}

# ---- 2x2: w in {3, 6} x goal in {true, lidar} ----
export LIDAR_ACT_CFG_W=3
export LIDAR_NAV_GOAL_MODE=true
if run_plan nav_true_w3; then peek rollout_results/00_rollout_nav_true_w3.json; fi
export LIDAR_NAV_GOAL_MODE=lidar
if run_plan nav_lidar_w3; then peek rollout_results/00_rollout_nav_lidar_w3.json; fi

export LIDAR_ACT_CFG_W=6
export LIDAR_NAV_GOAL_MODE=true
if run_plan nav_true_w6; then peek rollout_results/00_rollout_nav_true_w6.json; fi
export LIDAR_NAV_GOAL_MODE=lidar
if run_plan nav_lidar_w6; then peek rollout_results/00_rollout_nav_lidar_w6.json; fi

echo "=== DONE. READ: does nav_true_wN beat nav_lidar_wN (separation)? ==="
echo "  separation and/or mean min-dist << 48 -> channel usable: tune w, more eps,"
echo "      retrain still recommended for margin (h64_widek config, ready)."
echo "  no separation, means ~48 -> eval levers exhausted -> LAUNCH THE RETRAIN:"
echo "      sbatch train_lidar_invdyn.sbatch  (--config .../og_antM_Gi_lidar30_full_g30d_invdyn_h64_widek.py)"
echo "  w=6 much worse than w=3 (thrash/instability) -> amplifying noise; stop at w<=3."
