#!/bin/bash
# --- SLURM Settings ---
#SBATCH --job-name=lidar_noorient_50
#SBATCH --output=slurm_logs/lidar_noorient_50-%j.out
#SBATCH --error=slurm_logs/lidar_noorient_50-%j.out
#SBATCH --partition=part-preempt-classB
#SBATCH --qos=qos-preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00

# ============================================================================
# NO-ORIENTATION ABLATION (2026-08-20) of the 50/100 config, in an isolated
# worktree at commit f014b3f (fingerprint-fix-only, PERIODIC replanning, no
# dist_check threshold -- the exact commit/config behind the
# "localization-fix-7/20-and-50%-multi" branch). Identical launch recipe to
# that 50% run:
#   LIDAR_VERIFY_TOL=1.5 LIDAR_GOAL_TOL=1.0 LIDAR_FP_RANK=1 N_EP=100 --pl_seeds 0
# ONE swap: LIDAR_FP_RANK=1 -> LIDAR_NOORIENT_RANK=1 (LIDAR_FP_RANK=0) --
# goal-cell ranking now scores each candidate by its BEST match over ANY of
# its own headings to a single goal scan (the "no orientation" left panel
# from the similarity-map comparison), instead of the 12-heading
# index-aligned fingerprint. New method: _noorient_rank in
# ogb_stgl_sml_lidar_planner_v1.py (same patch as the 46%-config ablation).
# ============================================================================

mkdir -p slurm_logs rollout_results
source ~/.bashrc
source /home/projects/crml-prj10233/miniforge3/etc/profile.d/conda.sh
conda activate compdfu_ogb_release
export PATH="$CONDA_PREFIX/bin:$PATH"
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

# ---- 12/20 execution stack (identical to the 50% run) ----
export LIDAR_REPLAN=1
export LIDAR_REPL_EVERY=500
export LIDAR_MAX_N_REPL=15
export LIDAR_N_MAX_STEPS=8000
export LIDAR_GOAL_YAW=0.0
export LIDAR_LOC_XYSTEP=1.0
export LIDAR_LOC_NTHETA=12
export LIDAR_LOC_MOTION_SIGMA=1.0
export LIDAR_NAV_STEP_WU=0.13
export LIDAR_PICK_TYPE=first
export LIDAR_NAV_MODE=bfs
export LIDAR_NAV_PURSUIT=1
export LIDAR_NAV_RIBBON=data
export LIDAR_NAV_LOOKAHEAD=16
export LIDAR_INV_MODEL_PATH=$CHUNK16_DIR
export LIDAR_INV_EPOCH=1600000
export LIDAR_TF_NPZ="$HOME/.ogbench/data/antmaze-giant-stitch-v0.npz"

# ---- pure-LiDAR goal + self-declared arrival ----
export LIDAR_NAV_GOAL_MODE=lidar
export LIDAR_TF_BAND=1
export LIDAR_TF_BAND_STRIDE=16
export LIDAR_ARRIVAL=1
export LIDAR_ARRIVAL_ENDS=1
export LIDAR_ARRIVAL_GOAL_MODE=lidar

# ---- verify-and-hop ----
export LIDAR_NAV_TOPK=5
export LIDAR_VERIFY_HOP=1
export LIDAR_VERIFY_TOL=1.5
export LIDAR_VERIFY_MAX_HOPS=6
export LIDAR_VERIFY_MIN_CONF=0.0
export LIDAR_GOAL_TOL=1.0

# ---- NOTE: no LIDAR_REPLAN_MODE / LIDAR_ADA_DIST_THRES set -> defaults to
# ---- 'periodic' (no threshold), exactly the 50% run's config ----

# ---- THE ABLATION: no-orientation ranking instead of the 12-heading fingerprint ----
export LIDAR_NOORIENT_RANK=1
export LIDAR_FP_RANK=0

echo "=== NO-ORIENTATION ABLATION CONFIG (vs 50/100 periodic baseline) ==="
env | grep '^LIDAR_' | sort
echo "CONFIG=$CONFIG"
echo "===================="

N_EP=${N_EP:-100}
STAMP=$(mktemp); touch "$STAMP"
echo "=== [noorient-ablation-50] vs 50/100 (fingerprint rank, periodic) baseline, $N_EP ep, seed 0 ==="
PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=offline \
python diffuser/ogb_task/ogb_maze_v1/plan_ogb_stgl_sml.py \
    --config "$CONFIG" --plan_n_ep "$N_EP" --pl_seeds 0
CFG_TAG=$(basename "$CONFIG" .py)
SRC=$(find "$PLANS_DIR" -path "*${CFG_TAG}*" -name 00_rollout.json -newer "$STAMP" 2>/dev/null | head -1)
rm -f "$STAMP"
if [ -z "$SRC" ]; then echo '!! no rollout json found'; exit 1; fi
OUT_TAG="00_rollout_noorient_ablation_50.json"
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
print("  vs fingerprint rank (orientation-aware, periodic) baseline: 50/100")
PY

echo "=== DONE ==="
