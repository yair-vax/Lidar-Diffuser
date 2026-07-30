#!/bin/bash
# --- SLURM Settings ---
#SBATCH --job-name=lidar_wdrecall
#SBATCH --output=slurm_logs/lidar_wdrecall-%j.out
#SBATCH --error=slurm_logs/lidar_wdrecall-%j.out
#SBATCH --partition=part-preempt
#SBATCH --qos=qos-preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --exclude=ece-ykasten1

# ============================================================================
# WALKING-DISTANCE GOAL PRIOR, PRE-TRUNCATION -- the RECALL fix (2026-07-14).
#
# History: plan_lidar_walkdist_probe.sbatch (alpha=1.0) applied the walking-
# distance prior AFTER _goal_posterior_cells had already truncated to its
# top ~7 candidates by scan similarity alone. Result: 3/20 (up from 2/20),
# and true-cell rank=0 nearly doubled (4->9 of 20) -- but the true cell was
# STILL missing from the candidate list entirely in 10/20 episodes, exactly
# as before. Reordering a shortlist can't rescue a cell the shortlist itself
# never contained.
#
# The user's fix: compute a COMBINED score (scan similarity x walking-
# distance factor) for EVERY free cell in the maze FIRST, and only THEN pick
# the top-K shortlist from that combined ranking -- not top-K by scan alone,
# reweighted after. Implemented in _goal_posterior_cells (now takes
# dist_map/walkdist_alpha and reweights `mass` for the full per-cell array
# before the np.argsort(...)[:n_keep] truncation, instead of after).
#
# Offline-verified before spending GPU time (15 real dataset frames near the
# true goal, sampled): of 4 that were raw-LiDAR '>K' (scan similarity alone
# ranked the true cell outside top-7), alpha=1.0 rescued 2/4 into the
# candidate list; alpha=1.5 rescued all 4/4, with NO regression on the cases
# that already worked (rank stayed 0). Multiplicative form means a cell with
# near-zero scan similarity stays near-zero combined score regardless of
# alpha -- distance alone can't manufacture a contender, per the user's
# explicit ask ("far points which are not similar at all will not be
# contenders").
#
# READ:
#   >K rate drops meaningfully below 10/20 (50%) -> the recall fix is working;
#     srate should climb further as a result (compare to the 3/20 rerank-only
#     result and the 2/20 alpha=0 baseline).
#   >K rate barely moves -> either alpha=1.5 undershoots in the closed loop
#     the way it didn't offline (real episodes differ ep-to-ep; the 15-frame
#     offline sample is not all 20 actual episodes), or grid-quantization /
#     structural aliasing is severe enough that even full-maze scan mass is
#     too flat/wrong to rescue with a distance prior alone.
# ============================================================================

mkdir -p slurm_logs rollout_results
source ~/.bashrc
source /home/projects/crml-prj10233/miniforge3/etc/profile.d/conda.sh
conda activate compdfu_ogb_release
export PATH="$CONDA_PREFIX/bin:$PATH"   # guard vs stray PATH entries shadowing the env's python
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
echo "Job started on $(hostname)"
echo "python: $(which python) ($(python -c 'import torch; print("torch", torch.__version__)' 2>&1))"

CONFIG=${CONFIG:-config/ogb_ant_maze/og_antM_Gi_lidar30_Cd_Stgl_PadBuf_Ft64_ts512.py}
PLANS_DIR="logs/antmaze-giant-stitch-v0/plans"
CHUNK16_DIR="logs/antmaze-giant-stitch-v0/diffusion/og_antM_Gi_lidar30_full_g30d_invdyn_h24_chunk16"

find . -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null
find . -name '*.pyc' -delete 2>/dev/null

python -m py_compile diffuser/ogb_task/ogb_maze_v1/ogb_stgl_sml_lidar_planner_v1.py \
                     diffuser/ogb_task/ogb_maze_v1/ogb_stgl_sml_planner_v1.py \
                     diffuser/ogb_task/ogb_maze_v1/plan_ogb_stgl_sml.py \
                     diffuser/datasets/lidar/lidar_route_nav.py \
  || { echo '!! py_compile failed -- stale sync?'; exit 1; }

export LIDAR_LOC_NTHETA=${LIDAR_LOC_NTHETA:-12}   # original resolution -- isolate this one variable

python -c "from diffuser.datasets.lidar.lidar_route_nav import RouteScanSynthesizer; \
from diffuser.datasets.lidar.lidar_grid import load_or_build_grid; \
import os; \
load_or_build_grid('antmaze-giant-stitch-v0', dict(n_beams=30, max_range=12.0), xy_step=1.0, n_theta=int(os.environ['LIDAR_LOC_NTHETA']))" \
  || { echo '!! route-nav import / grid build failed'; exit 1; }

# ---- 12/20 execution stack (identical to the bench defaults) ----
export LIDAR_REPLAN=${LIDAR_REPLAN:-1}
export LIDAR_REPL_EVERY=${LIDAR_REPL_EVERY:-500}
export LIDAR_MAX_N_REPL=${LIDAR_MAX_N_REPL:-15}
export LIDAR_N_MAX_STEPS=${LIDAR_N_MAX_STEPS:-8000}
export LIDAR_GOAL_YAW=${LIDAR_GOAL_YAW:-0.0}
export LIDAR_LOC_XYSTEP=${LIDAR_LOC_XYSTEP:-1.0}
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

# ---- the swapped rung: pure-LiDAR goal + self-declared arrival ----
export LIDAR_NAV_GOAL_MODE=lidar
export LIDAR_TF_BAND=1
export LIDAR_TF_BAND_STRIDE=${LIDAR_TF_BAND_STRIDE:-16}
export LIDAR_ARRIVAL=1
export LIDAR_ARRIVAL_ENDS=1
export LIDAR_ARRIVAL_GOAL_MODE=lidar

# ---- verify-and-hop: best known config so far (single-scan, tol=1.5) ----
export LIDAR_NAV_TOPK=${LIDAR_NAV_TOPK:-5}
export LIDAR_VERIFY_HOP=1
export LIDAR_VERIFY_TOL=${LIDAR_VERIFY_TOL:-1.5}
export LIDAR_VERIFY_MAX_HOPS=${LIDAR_VERIFY_MAX_HOPS:-6}
export LIDAR_VERIFY_MIN_CONF=${LIDAR_VERIFY_MIN_CONF:-0.0}
export LIDAR_VERIFY_SEQ_LEN=${LIDAR_VERIFY_SEQ_LEN:-1}

# ---- walking-distance goal prior, now applied PRE-truncation (this probe's whole point) ----
export LIDAR_WALKDIST_ALPHA=${LIDAR_WALKDIST_ALPHA:-1.5}

echo "=== PROBE CONFIG ==="
env | grep '^LIDAR_' | sort
echo "CONFIG=$CONFIG"
echo "===================="

N_EP=${N_EP:-20}
STAMP=$(mktemp); touch "$STAMP"
echo "=== [walkdist-recall] GOAL=lidar+tf_band ARRIVAL=1 VERIFY_HOP=1 WALKDIST_ALPHA=$LIDAR_WALKDIST_ALPHA (pre-truncation) TOL=$LIDAR_VERIFY_TOL TOPK=$LIDAR_NAV_TOPK RIBBON=$LIDAR_NAV_RIBBON LOOKAHEAD=$LIDAR_NAV_LOOKAHEAD STEPS=$LIDAR_N_MAX_STEPS REPL=$LIDAR_REPL_EVERY, $N_EP ep ==="
PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=offline \
python diffuser/ogb_task/ogb_maze_v1/plan_ogb_stgl_sml.py \
    --config "$CONFIG" --plan_n_ep "$N_EP" --pl_seeds 0
CFG_TAG=$(basename "$CONFIG" .py)
SRC=$(find "$PLANS_DIR" -path "*${CFG_TAG}*" -name 00_rollout.json -newer "$STAMP" 2>/dev/null | head -1)
rm -f "$STAMP"
if [ -z "$SRC" ]; then echo '!! no rollout json found'; exit 1; fi
OUT_TAG="00_rollout_walkdist_recall_alpha${LIDAR_WALKDIST_ALPHA}.json"
cp "$SRC" "rollout_results/${OUT_TAG}"
echo "saved -> rollout_results/${OUT_TAG}"

python - "$OUT_TAG" <<'PY'
import json, sys
out_tag = sys.argv[1]
d = json.load(open(f'rollout_results/{out_tag}'))
s = [bool(x) for x in d.get('ep_is_suc', [])]
m = d.get('ep_min_goal_dist', [])
print(f"[peek] {out_tag}: {sum(s)}/{len(s)}  srate={d.get('ep_srate')}")
if m: print(f"  min_goal_dist best={min(m):.1f} mean={sum(m)/len(m):.1f} worst={max(m):.1f} mj (tol 0.5)")
print("  vs walkdist rerank-only alpha=1.0 (post-truncation): 3/20")
print("  vs walkdist_alpha=0 (verify-hop tol=1.5, n_theta=12): 2/20")
print("  vs n_theta=36 (no walkdist prior): 2/20")
print("  vs puregoal_tf baseline (verify-hop OFF): 0/20")
print("  vs nav_true baseline @8000: 12/20, median arrival 4192")
PY

echo "=== check true-cell rank distribution this run ==="
grep "gridnav/topk" "slurm_logs/lidar_wdrecall-${SLURM_JOB_ID}.out" | grep -oP "rank=\K\S+" | sort | uniq -c

echo "=== DONE ==="
