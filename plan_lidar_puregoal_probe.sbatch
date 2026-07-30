#!/bin/bash
# --- SLURM Settings ---
#SBATCH --job-name=lidar_puregoal
#SBATCH --output=slurm_logs/lidar_puregoal-%j.out
#SBATCH --error=slurm_logs/lidar_puregoal-%j.out
#SBATCH --partition=part-preempt
#SBATCH --qos=qos-preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --exclude=ece-ykasten1

# ============================================================================
# PURE-LiDAR-GOAL PROBE (2026-07-08) -- the spec-headline arm, 20 ep, seed 0.
#
# Same execution stack as the 12/20 run (chunk16 x A2 data ribbon x repl500 x
# 8000 steps, pursuit v2) with ONE rung swapped: the goal cell is NOT given --
# it is LOCALIZED from a REAL goal scan band (LIDAR_TF_BAND=1, stride 16, the
# gridnav arm-3 recipe) and the ant SELF-DECLARES arrival (LIDAR_ARRIVAL=1,
# ARRIVAL_ENDS=1). ep_srate stays ground-truth env success, so:
#   - goal-localization error: grep '[gridnav] ep goal .* cells off' (history:
#     real band exact cell in 5/20; synth band decoys 8-11 cells off 20/20)
#   - arrival false positives: episodes that END on arrival with is_suc=False
#
# READ vs nav_true 12/20 @8000:
#   ~10-12/20 -> goal rung ~solved; bench100 THIS arm next (the spec headline).
#   ~3-6/20 with successes exactly where 'cells off'==0 -> executor fine, goal
#     localization is the last rung; work the band/filter (not the executor).
#   ~0 despite some 'cells off'==0 -> arrival/stop logic suspect: compare
#     ep_min_goal_dist of the 0-off episodes vs their end cause.
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
CHUNK16_DIR="logs/antmaze-giant-stitch-v0/diffusion/og_antM_Gi_lidar30_full_g30d_invdyn_h24_chunk16"

find . -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null
find . -name '*.pyc' -delete 2>/dev/null

python -m py_compile diffuser/ogb_task/ogb_maze_v1/ogb_stgl_sml_lidar_planner_v1.py \
                     diffuser/ogb_task/ogb_maze_v1/ogb_stgl_sml_planner_v1.py \
                     diffuser/ogb_task/ogb_maze_v1/plan_ogb_stgl_sml.py \
  || { echo '!! py_compile failed -- stale sync?'; exit 1; }
python -c "from diffuser.datasets.lidar.lidar_route_nav import RouteScanSynthesizer; \
from diffuser.datasets.lidar.lidar_grid import load_or_build_grid; \
load_or_build_grid('antmaze-giant-stitch-v0', dict(n_beams=30, max_range=12.0), xy_step=1.0, n_theta=12)" \
  || { echo '!! route-nav import / grid build failed'; exit 1; }

# ---- 12/20 execution stack (identical to the bench defaults) ----
export LIDAR_REPLAN=${LIDAR_REPLAN:-1}
export LIDAR_REPL_EVERY=${LIDAR_REPL_EVERY:-500}
export LIDAR_MAX_N_REPL=${LIDAR_MAX_N_REPL:-15}
export LIDAR_N_MAX_STEPS=${LIDAR_N_MAX_STEPS:-8000}
export LIDAR_GOAL_YAW=${LIDAR_GOAL_YAW:-0.0}
export LIDAR_LOC_XYSTEP=${LIDAR_LOC_XYSTEP:-1.0}
export LIDAR_LOC_NTHETA=${LIDAR_LOC_NTHETA:-12}
export LIDAR_LOC_MOTION_SIGMA=${LIDAR_LOC_MOTION_SIGMA:-1.0}
export LIDAR_NAV_STEP_WU=${LIDAR_NAV_STEP_WU:-0.13}
export LIDAR_PICK_TYPE=${LIDAR_PICK_TYPE:-first}
export LIDAR_NAV_MODE=${LIDAR_NAV_MODE:-bfs}
export LIDAR_NAV_PURSUIT=${LIDAR_NAV_PURSUIT:-1}
export LIDAR_NAV_RIBBON=${LIDAR_NAV_RIBBON:-data}
export LIDAR_NAV_LOOKAHEAD=${LIDAR_NAV_LOOKAHEAD:-16}   # match whatever the 12/20 job used
export LIDAR_INV_MODEL_PATH=${LIDAR_INV_MODEL_PATH:-$CHUNK16_DIR}
export LIDAR_INV_EPOCH=${LIDAR_INV_EPOCH:-1600000}
export LIDAR_TF_NPZ=${LIDAR_TF_NPZ:-"$HOME/.ogbench/data/antmaze-giant-stitch-v0.npz"}

# ---- the ONE swapped rung: pure-LiDAR goal + self-declared arrival ----
export LIDAR_NAV_GOAL_MODE=lidar
export LIDAR_TF_BAND=1
export LIDAR_TF_BAND_STRIDE=${LIDAR_TF_BAND_STRIDE:-16}
export LIDAR_ARRIVAL=1
export LIDAR_ARRIVAL_ENDS=1
export LIDAR_ARRIVAL_GOAL_MODE=lidar

echo "=== PROBE CONFIG ==="
env | grep '^LIDAR_' | sort
echo "CONFIG=$CONFIG"
echo "===================="

N_EP=${N_EP:-20}
STAMP=$(mktemp); touch "$STAMP"
echo "=== [puregoal_tf] GOAL=lidar+tf_band ARRIVAL=1 RIBBON=$LIDAR_NAV_RIBBON LOOKAHEAD=$LIDAR_NAV_LOOKAHEAD STEPS=$LIDAR_N_MAX_STEPS REPL=$LIDAR_REPL_EVERY, $N_EP ep ==="
PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=offline \
python diffuser/ogb_task/ogb_maze_v1/plan_ogb_stgl_sml.py \
    --config "$CONFIG" --plan_n_ep "$N_EP" --pl_seeds 0
CFG_TAG=$(basename "$CONFIG" .py)
SRC=$(find "$PLANS_DIR" -path "*${CFG_TAG}*" -name 00_rollout.json -newer "$STAMP" 2>/dev/null | head -1)
rm -f "$STAMP"
if [ -z "$SRC" ]; then echo '!! no rollout json found'; exit 1; fi
cp "$SRC" "rollout_results/00_rollout_puregoal_tf.json"
echo "saved -> rollout_results/00_rollout_puregoal_tf.json"

python - <<'PY'
import json
d = json.load(open('rollout_results/00_rollout_puregoal_tf.json'))
s = [bool(x) for x in d.get('ep_is_suc', [])]
m = d.get('ep_min_goal_dist', [])
print(f"[peek] puregoal_tf: {sum(s)}/{len(s)}  srate={d.get('ep_srate')}")
if m: print(f"  min_goal_dist best={min(m):.1f} mean={sum(m)/len(m):.1f} worst={max(m):.1f} mj (tol 0.5)")
print("  vs nav_true baseline @8000: 12/20, median arrival 4192")
print("  NEXT: grep this log for '\\[gridnav\\] ep goal' -> cells-off per episode;")
print("        cross e.g. cells-off==0 episodes vs their is_suc.")
PY

echo "=== DONE ==="
