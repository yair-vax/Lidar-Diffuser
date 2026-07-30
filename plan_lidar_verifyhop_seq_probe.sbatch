#!/bin/bash
# --- SLURM Settings ---
#SBATCH --job-name=lidar_verifyseq
#SBATCH --output=slurm_logs/lidar_verifyseq-%j.out
#SBATCH --error=slurm_logs/lidar_verifyseq-%j.out
#SBATCH --partition=part-preempt
#SBATCH --qos=qos-preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --exclude=ece-ykasten1

# ============================================================================
# PURE-LiDAR-GOAL PROBE + VERIFY-AND-HOP, SEQUENCE MATCH (2026-07-14).
#
# History: plan_lidar_puregoal_probe.sbatch (verify-hop off) scored 0/20.
# plan_lidar_verifyhop_probe.sbatch turned verify-and-hop ON:
#   - tol=1.5: 2/20, but 0 ACCEPTs / 37 REJECTs -- the mechanism never once
#     confirmed a match; successes came from the env's ground-truth check
#     firing while wandering hop targets, not from the mechanism itself.
#   - tol=2.5: 1/20 (WORSE) -- and it revealed WHY tightening/loosening a
#     single-scan threshold can't work here: two WRONG cells (16 and several
#     cells off the true goal) scored a BETTER (lower) live-vs-goal RMS than
#     the TRUE goal cell did (1.975, 2.137 vs 2.246). A single instantaneous
#     LiDAR scan aliases: some other corridor/room in the maze can look more
#     like the goal, by chance, than the goal itself does at that instant.
#
# This run adds LIDAR_VERIFY_SEQ_LEN (new code, ogb_stgl_sml_lidar_planner_v1.py
# _verify_arrival): instead of comparing ONE live scan to ONE goal scan, it
# compares a WINDOW of the most recent live scans to the matching tail frames
# of the goal band (`_goal_scan_band`, already rendered as a multi-step
# approach into the goal -- previously only its LAST frame was ever used).
# seq_len=4 uses the full band (goal_band_k=4 in this config, confirmed via
# `[tf_band] ... band (4, 30)` in the prior logs). Tolerance is put BACK to
# the strict 1.5 (not 2.5) so this is a clean test of whether sequence
# matching fixes the aliasing itself, rather than just loosening further.
#
# READ:
#   ACCEPTs appear now, and skew towards the TRUE cell / better srate than
#     both prior runs -> multi-frame matching resolves the aliasing; scale it
#     up (bigger goal_band_k, retrain with more approach context)
#   still 0 ACCEPTs / no improvement -> even a 4-frame window aliases just as
#     badly (the decoy corridors are themselves multi-cell-long look-alikes,
#     not one-off coincidences) -- the fix has to be richer LiDAR (more beams
#     / longer range) or a non-LiDAR disambiguator, not more temporal context
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

# ---- verify-and-hop, SEQUENCE match (this probe's whole point) ----
export LIDAR_NAV_TOPK=${LIDAR_NAV_TOPK:-5}
export LIDAR_VERIFY_HOP=1
export LIDAR_VERIFY_TOL=${LIDAR_VERIFY_TOL:-1.5}
export LIDAR_VERIFY_MAX_HOPS=${LIDAR_VERIFY_MAX_HOPS:-6}
export LIDAR_VERIFY_MIN_CONF=${LIDAR_VERIFY_MIN_CONF:-0.0}
export LIDAR_VERIFY_SEQ_LEN=${LIDAR_VERIFY_SEQ_LEN:-4}

echo "=== PROBE CONFIG ==="
env | grep '^LIDAR_' | sort
echo "CONFIG=$CONFIG"
echo "===================="

N_EP=${N_EP:-20}
STAMP=$(mktemp); touch "$STAMP"
echo "=== [verifyseq] GOAL=lidar+tf_band ARRIVAL=1 VERIFY_HOP=1 SEQ_LEN=$LIDAR_VERIFY_SEQ_LEN TOPK=$LIDAR_NAV_TOPK TOL=$LIDAR_VERIFY_TOL MAX_HOPS=$LIDAR_VERIFY_MAX_HOPS RIBBON=$LIDAR_NAV_RIBBON LOOKAHEAD=$LIDAR_NAV_LOOKAHEAD STEPS=$LIDAR_N_MAX_STEPS REPL=$LIDAR_REPL_EVERY, $N_EP ep ==="
PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=offline \
python diffuser/ogb_task/ogb_maze_v1/plan_ogb_stgl_sml.py \
    --config "$CONFIG" --plan_n_ep "$N_EP" --pl_seeds 0
CFG_TAG=$(basename "$CONFIG" .py)
SRC=$(find "$PLANS_DIR" -path "*${CFG_TAG}*" -name 00_rollout.json -newer "$STAMP" 2>/dev/null | head -1)
rm -f "$STAMP"
if [ -z "$SRC" ]; then echo '!! no rollout json found'; exit 1; fi
OUT_TAG="00_rollout_verifyseq_len${LIDAR_VERIFY_SEQ_LEN}_tol${LIDAR_VERIFY_TOL}.json"
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
print("  vs verify-hop single-scan tol=1.5: 2/20")
print("  vs verify-hop single-scan tol=2.5: 1/20")
print("  vs puregoal_tf baseline (verify-hop OFF): 0/20")
print("  vs nav_true baseline @8000: 12/20, median arrival 4192")
PY

echo "=== DONE ==="
