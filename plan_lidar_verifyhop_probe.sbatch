#!/bin/bash
# --- SLURM Settings ---
#SBATCH --job-name=lidar_verifyhop
#SBATCH --output=slurm_logs/lidar_verifyhop-%j.out
#SBATCH --error=slurm_logs/lidar_verifyhop-%j.out
#SBATCH --partition=part-preempt
#SBATCH --qos=qos-preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --exclude=ece-ykasten1

# ============================================================================
# PURE-LiDAR-GOAL PROBE + VERIFY-AND-HOP (2026-07-14).
#
# Identical execution stack to plan_lidar_puregoal_probe.sbatch (which scored
# 0/20: goal-posterior top5 often didn't rank the true cell first, and the
# self-declared arrival trusted whatever cell it first localized to).
#
# The ONE thing swapped here: verify-and-hop is turned ON (LIDAR_VERIFY_HOP=1).
# This is EXISTING code in ogb_stgl_sml_lidar_planner_v1.py
# (_verify_arrival / _nav_hop_to_next, "memory 2026-07-08 fix-1") that was
# already built but never exercised by any script -- every prior run left it
# at its default-off value. It works off the top-K goal-posterior list that
# is already computed and logged every episode (`[gridnav/topk]`):
#   - on a self-declared arrival, compare the live scan to the goal band's
#     tail scan (the fixed, pure-LiDAR goal descriptor)
#   - match (<= LIDAR_VERIFY_TOL per-beam RMS wu) -> accept, episode ends
#   - mismatch -> reject this cell, hop to the next-best posterior hypothesis
#     (up to LIDAR_VERIFY_MAX_HOPS times), instead of parking on hypothesis #1
#
# READ:
#   improvement over the 0/20 puregoal_tf baseline -> the top-K list does
#     contain better hypotheses than the argmax; verify-and-hop retrieves them
#   no improvement -> even the top-K family misses the true cell too often
#     (see true_rank in `[gridnav/topk]` lines) -- localization itself, not
#     the commit strategy, is the bottleneck; revisit richer LiDAR obs instead
#
# UPDATE (2026-07-14, tol=1.5 run): scored 2/20 (up from 0/20), but the
# verify-and-hop ACCEPT/REJECT log showed 37 REJECTs and ZERO ACCEPTs across
# the whole run -- every self-declared arrival's live-vs-goal scan RMS came in
# above tol=1.5 (min rejected value: 1.975), so the 2 successes came from the
# env's ground-truth check firing while wandering hop targets, not from the
# mechanism ever confirming a match. This run bumps LIDAR_VERIFY_TOL to 2.5
# (chosen from a gap in that reject distribution: 4 rejects clustered at
# 1.975-2.246, then a jump to 2.764+) to test whether the tolerance itself
# was simply mis-set too strict for this scan-matching noise floor.
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

# ---- the swapped rung: pure-LiDAR goal + self-declared arrival ----
export LIDAR_NAV_GOAL_MODE=lidar
export LIDAR_TF_BAND=1
export LIDAR_TF_BAND_STRIDE=${LIDAR_TF_BAND_STRIDE:-16}
export LIDAR_ARRIVAL=1
export LIDAR_ARRIVAL_ENDS=1
export LIDAR_ARRIVAL_GOAL_MODE=lidar

# ---- NEW: verify-and-hop turned ON (this probe's whole point) ----
export LIDAR_NAV_TOPK=${LIDAR_NAV_TOPK:-5}
export LIDAR_VERIFY_HOP=1
export LIDAR_VERIFY_TOL=${LIDAR_VERIFY_TOL:-2.5}
export LIDAR_VERIFY_MAX_HOPS=${LIDAR_VERIFY_MAX_HOPS:-6}
export LIDAR_VERIFY_MIN_CONF=${LIDAR_VERIFY_MIN_CONF:-0.0}

echo "=== PROBE CONFIG ==="
env | grep '^LIDAR_' | sort
echo "CONFIG=$CONFIG"
echo "===================="

N_EP=${N_EP:-20}
STAMP=$(mktemp); touch "$STAMP"
echo "=== [verifyhop] GOAL=lidar+tf_band ARRIVAL=1 VERIFY_HOP=1 TOPK=$LIDAR_NAV_TOPK TOL=$LIDAR_VERIFY_TOL MAX_HOPS=$LIDAR_VERIFY_MAX_HOPS RIBBON=$LIDAR_NAV_RIBBON LOOKAHEAD=$LIDAR_NAV_LOOKAHEAD STEPS=$LIDAR_N_MAX_STEPS REPL=$LIDAR_REPL_EVERY, $N_EP ep ==="
PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=offline \
python diffuser/ogb_task/ogb_maze_v1/plan_ogb_stgl_sml.py \
    --config "$CONFIG" --plan_n_ep "$N_EP" --pl_seeds 0
CFG_TAG=$(basename "$CONFIG" .py)
SRC=$(find "$PLANS_DIR" -path "*${CFG_TAG}*" -name 00_rollout.json -newer "$STAMP" 2>/dev/null | head -1)
rm -f "$STAMP"
if [ -z "$SRC" ]; then echo '!! no rollout json found'; exit 1; fi
OUT_TAG="00_rollout_verifyhop_tol${LIDAR_VERIFY_TOL}.json"
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
print("  vs verify-hop tol=1.5: 2/20")
print("  vs puregoal_tf baseline (verify-hop OFF): 0/20")
print("  vs nav_true baseline @8000: 12/20, median arrival 4192")
PY

echo "=== DONE ==="
