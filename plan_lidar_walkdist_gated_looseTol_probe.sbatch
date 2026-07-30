#!/bin/bash
# --- SLURM Settings ---
#SBATCH --job-name=lidar_wdlooseT
#SBATCH --output=slurm_logs/lidar_wdlooseT-%j.out
#SBATCH --error=slurm_logs/lidar_wdlooseT-%j.out
#SBATCH --partition=part-preempt
#SBATCH --qos=qos-preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --exclude=ece-ykasten1

# ============================================================================
# WALKDIST PRIOR + GATE, WITH LOOSENED VERIFY TOLERANCE (2026-07-14).
#
# Just found, in the walkdist rerank-only run (alpha=1.0, job 26114, 3/20):
# episodes 12 and 16 BOTH got the goal-cell pick exactly right on the first
# try (walkdist prior correctly identified true cell (10,14), 0 cells off) --
# but verify-hop's tolerance check then REJECTED that correct arrival anyway
# (live-vs-goal scan RMS 2.246 and 4.764, both > tol=1.5) and hopped the ant
# away to a WRONG cell, dragging it from a close approach (min_goal_dist 0.66,
# 0.97) out to final_goal_dist 9.5 and 11.3. The prior is now good enough that
# verify-hop's old, strict tolerance is actively sabotaging correct answers,
# not just failing to confirm ambiguous ones.
#
# This run keeps the current best identification stack (walkdist_alpha=1.5 +
# similarity gate 0.15, matching plan_lidar_walkdist_gated_probe.sbatch,
# pending its own closed-loop result) and ONLY loosens LIDAR_VERIFY_TOL to
# 5.0 -- comfortably above both known true-positive rejects (2.246, 4.764) --
# to test whether trusting the (now much better-curated) top candidate more
# recovers those near-misses instead of hopping away from them.
#
# Caveat carried over from the earlier tol=2.5 experiment (back when there was
# no walkdist prior at all): looser tolerance previously let false positives
# through elsewhere. This is a genuinely different test, not a repeat -- the
# CANDIDATE SET being hopped through is now walkdist+gate-curated, so the
# earlier false-positive risk profile may not transfer directly. That is
# exactly what this run checks.
#
# READ:
#   srate clears 3/20 with no new false ACCEPTs at implausible cells -> the
#     tolerance really was the last blocker on an already-good identification
#     stack; keep tol loose going forward.
#   new false ACCEPTs appear at cells the walkdist prior/gate let through ->
#     tolerance still needs to discriminate WITHIN the curated candidate set,
#     not just be loosened uniformly; consider a tol that scales with a
#     candidate's own combined score/confidence instead of one fixed number.
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

# ---- verify-and-hop, LOOSENED tolerance (this probe's whole point) ----
export LIDAR_NAV_TOPK=${LIDAR_NAV_TOPK:-5}
export LIDAR_VERIFY_HOP=1
export LIDAR_VERIFY_TOL=${LIDAR_VERIFY_TOL:-5.0}
export LIDAR_VERIFY_MAX_HOPS=${LIDAR_VERIFY_MAX_HOPS:-6}
export LIDAR_VERIFY_MIN_CONF=${LIDAR_VERIFY_MIN_CONF:-0.0}
export LIDAR_VERIFY_SEQ_LEN=${LIDAR_VERIFY_SEQ_LEN:-1}

# ---- walking-distance goal prior, pre-truncation, WITH the hard similarity gate ----
export LIDAR_WALKDIST_ALPHA=${LIDAR_WALKDIST_ALPHA:-1.5}
export LIDAR_WALKDIST_MIN_SIM_FRAC=${LIDAR_WALKDIST_MIN_SIM_FRAC:-0.15}

echo "=== PROBE CONFIG ==="
env | grep '^LIDAR_' | sort
echo "CONFIG=$CONFIG"
echo "===================="

N_EP=${N_EP:-20}
STAMP=$(mktemp); touch "$STAMP"
echo "=== [walkdist-gated-looseTol] GOAL=lidar+tf_band ARRIVAL=1 VERIFY_HOP=1 TOL=$LIDAR_VERIFY_TOL WALKDIST_ALPHA=$LIDAR_WALKDIST_ALPHA MIN_SIM_FRAC=$LIDAR_WALKDIST_MIN_SIM_FRAC TOPK=$LIDAR_NAV_TOPK RIBBON=$LIDAR_NAV_RIBBON LOOKAHEAD=$LIDAR_NAV_LOOKAHEAD STEPS=$LIDAR_N_MAX_STEPS REPL=$LIDAR_REPL_EVERY, $N_EP ep ==="
PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=offline \
python diffuser/ogb_task/ogb_maze_v1/plan_ogb_stgl_sml.py \
    --config "$CONFIG" --plan_n_ep "$N_EP" --pl_seeds 0
CFG_TAG=$(basename "$CONFIG" .py)
SRC=$(find "$PLANS_DIR" -path "*${CFG_TAG}*" -name 00_rollout.json -newer "$STAMP" 2>/dev/null | head -1)
rm -f "$STAMP"
if [ -z "$SRC" ]; then echo '!! no rollout json found'; exit 1; fi
OUT_TAG="00_rollout_walkdist_gated_tol${LIDAR_VERIFY_TOL}.json"
cp "$SRC" "rollout_results/${OUT_TAG}"
echo "saved -> rollout_results/${OUT_TAG}"

python - "$OUT_TAG" <<'PY'
import json, sys
out_tag = sys.argv[1]
d = json.load(open(f'rollout_results/{out_tag}'))
s = [bool(x) for x in d.get('ep_is_suc', [])]
m = d.get('ep_min_goal_dist', [])
f = d.get('ep_final_goal_dist', [])
print(f"[peek] {out_tag}: {sum(s)}/{len(s)}  srate={d.get('ep_srate')}")
if m: print(f"  min_goal_dist best={min(m):.1f} mean={sum(m)/len(m):.1f} worst={max(m):.1f} mj (tol 0.5)")
if m and f:
    close_but_wandered = [i for i in range(len(m)) if m[i] < 1.0 and not s[i]]
    print(f"  episodes that got within 1.0 mj of goal but ended up far (min<1.0, final={[round(f[i],1) for i in close_but_wandered]}): {close_but_wandered}")
print("  vs walkdist rerank-only alpha=1.0, tol=1.5 (post-truncation): 3/20 (eps 12,16 were near-misses rejected+hopped away)")
print("  vs walkdist_alpha=0, tol=1.5: 2/20")
print("  vs puregoal_tf baseline (verify-hop OFF): 0/20")
print("  vs nav_true baseline @8000: 12/20, median arrival 4192")
PY

echo "=== ACCEPT/REJECT counts + true-cell rank distribution this run ==="
grep -c "verify-hop\] ACCEPT" "slurm_logs/lidar_wdlooseT-${SLURM_JOB_ID}.out"
grep -c "verify-hop\] REJECT" "slurm_logs/lidar_wdlooseT-${SLURM_JOB_ID}.out"
grep "gridnav/topk" "slurm_logs/lidar_wdlooseT-${SLURM_JOB_ID}.out" | grep -oP "rank=\K\S+" | sort | uniq -c

echo "=== DONE ==="
