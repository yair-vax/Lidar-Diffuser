#!/bin/bash
# --- SLURM Settings ---
#SBATCH --job-name=lidar_walkdist
#SBATCH --output=slurm_logs/lidar_walkdist-%j.out
#SBATCH --error=slurm_logs/lidar_walkdist-%j.out
#SBATCH --partition=part-preempt
#SBATCH --qos=qos-preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --exclude=ece-ykasten1

# ============================================================================
# WALKING-DISTANCE GOAL PRIOR (2026-07-14).
#
# History: single-scan verify-hop (tol=1.5) plateaued at 2/20, with 50% of
# episodes never even ranking the true goal cell as a candidate. Neither a
# looser tolerance, sequence matching, nor a 3x finer orientation grid moved
# that number -- and the finer grid actually made the >K rate worse (65%).
#
# New idea (not a localization-quality fix -- a PRIOR fix): this is a giant
# STITCH maze. Start/goal pairs are drawn far apart by construction. Verified
# directly: the true goal is 30 cells of walking distance (BFS over the maze
# occupancy grid, NOT aerial/straight-line) from the shared start used in
# these 20 episodes -- and EVERY decoy cell that has out-scored it on scan
# similarity alone is much closer (4-29 cells). Aerial distance doesn't
# capture this (some "close" decoys are aerially far but walking-near, and
# vice versa, because of walls) -- BFS walking distance is the right metric.
#
# New code (ogb_stgl_sml_lidar_planner_v1.py _nav_localize_goal +
# lidar_route_nav.bfs_dist_map): rescales each goal-posterior candidate's
# scan-similarity score by (walking_dist_from_start + 1) ** LIDAR_WALKDIST_ALPHA
# before ranking, and uses the reweighted top candidate as the goal pick
# (previously a separately-computed raw MAP pose that could silently disagree
# with the aggregated top hypothesis). alpha=0 is the exact prior behavior;
# this run uses alpha=1.0, verified offline to flip true-cell vs (5,3) from
# 1.80 vs 1.98 (loses) to 55.8 vs 13.9 (wins) on the one traced example.
#
# READ:
#   true_rank improves broadly / srate climbs above 2/20 -> the giant-maze
#     far-goal prior is doing real work; sweep alpha, consider baking a
#     similar prior into training (not just eval-time reweighting).
#   no improvement -> either the prior is too strong/weak for OTHER episodes'
#     start-goal geometry (sweep alpha down/up before abandoning), or scan
#     aliasing dominates cases where a decoy is ALSO far from start.
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

export LIDAR_LOC_NTHETA=${LIDAR_LOC_NTHETA:-12}   # back to the original resolution -- isolate this one variable

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

# ---- NEW: walking-distance goal prior (this probe's whole point) ----
export LIDAR_WALKDIST_ALPHA=${LIDAR_WALKDIST_ALPHA:-1.0}

echo "=== PROBE CONFIG ==="
env | grep '^LIDAR_' | sort
echo "CONFIG=$CONFIG"
echo "===================="

N_EP=${N_EP:-20}
STAMP=$(mktemp); touch "$STAMP"
echo "=== [walkdist] GOAL=lidar+tf_band ARRIVAL=1 VERIFY_HOP=1 WALKDIST_ALPHA=$LIDAR_WALKDIST_ALPHA TOL=$LIDAR_VERIFY_TOL TOPK=$LIDAR_NAV_TOPK RIBBON=$LIDAR_NAV_RIBBON LOOKAHEAD=$LIDAR_NAV_LOOKAHEAD STEPS=$LIDAR_N_MAX_STEPS REPL=$LIDAR_REPL_EVERY, $N_EP ep ==="
PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=offline \
python diffuser/ogb_task/ogb_maze_v1/plan_ogb_stgl_sml.py \
    --config "$CONFIG" --plan_n_ep "$N_EP" --pl_seeds 0
CFG_TAG=$(basename "$CONFIG" .py)
SRC=$(find "$PLANS_DIR" -path "*${CFG_TAG}*" -name 00_rollout.json -newer "$STAMP" 2>/dev/null | head -1)
rm -f "$STAMP"
if [ -z "$SRC" ]; then echo '!! no rollout json found'; exit 1; fi
OUT_TAG="00_rollout_walkdist_alpha${LIDAR_WALKDIST_ALPHA}.json"
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
print("  vs walkdist_alpha=0 (verify-hop tol=1.5, n_theta=12): 2/20")
print("  vs n_theta=36 (no walkdist prior): 2/20")
print("  vs puregoal_tf baseline (verify-hop OFF): 0/20")
print("  vs nav_true baseline @8000: 12/20, median arrival 4192")
PY

echo "=== check true-cell rank distribution this run ==="
grep "gridnav/topk" "slurm_logs/lidar_walkdist-${SLURM_JOB_ID}.out" | grep -oP "rank=\K\S+" | sort | uniq -c

echo "=== DONE ==="
