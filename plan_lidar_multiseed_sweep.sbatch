#!/bin/bash
# --- SLURM Settings ---
#SBATCH --job-name=lidar_sweep
#SBATCH --output=slurm_logs/lidar_sweep-%j.out
#SBATCH --error=slurm_logs/lidar_sweep-%j.out
#SBATCH --partition=part-preempt
#SBATCH --qos=qos-preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --exclude=ece-ykasten1

# ============================================================================
# FULL-BENCHMARK MULTI-SEED SWEEP -- fair baseline-vs-fix comparison (2026-07-21).
#
# We've now seen THREE times that a single seed per problem is too noisy to
# trust for comparing configs: the reject-persistence-only run kept the same
# 2/20 total but completely swapped WHICH two episodes succeeded; adding
# goal-refine on top dropped to 1/20, indistinguishable from noise given the
# first result. A multi-seed run of the FIXED config compared against the
# single-seed BASELINE number (2/20) would be just as unfair -- that baseline
# is exactly as noisy as everything else. This runs ALL 20 problems x
# LIDAR_N_SEEDS seeds EACH for whichever config is selected (baseline or
# fixed, via the env vars below) -- run this script TWICE, once per config,
# to get two directly comparable multi-seed numbers.
#
# Uses the general multi-seed branch in plan_ogb_stgl_sml.py main() (plan_n_ep
# now runs the FULL 20-problem benchmark per seed call, not just one episode
# -- generalized from the single-episode ep12 heatmap harness). Reuses the
# already-loaded model across all seeds. Saves per-seed, per-problem
# ep_is_suc/ep_min_goal_dist to LIDAR_HEATMAP_OUT.
#
# Usage:
#   LIDAR_HEATMAP_OUT=rollout_results/sweep_baseline.json sbatch plan_lidar_multiseed_sweep.sbatch
#   LIDAR_VERIFY_REJECT_PERSIST=3 LIDAR_GOAL_REFINE=1 \
#     LIDAR_HEATMAP_OUT=rollout_results/sweep_fixed.json sbatch plan_lidar_multiseed_sweep.sbatch
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
CHUNK16_DIR="logs/antmaze-giant-stitch-v0/diffusion/og_antM_Gi_lidar30_full_g30d_invdyn_h24_chunk16"

find . -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null
find . -name '*.pyc' -delete 2>/dev/null

python -m py_compile diffuser/ogb_task/ogb_maze_v1/ogb_stgl_sml_lidar_planner_v1.py \
                     diffuser/ogb_task/ogb_maze_v1/ogb_stgl_sml_planner_v1.py \
                     diffuser/ogb_task/ogb_maze_v1/plan_ogb_stgl_sml.py \
  || { echo '!! py_compile failed -- stale sync?'; exit 1; }

export LIDAR_LOC_NTHETA=${LIDAR_LOC_NTHETA:-12}

python -c "from diffuser.datasets.lidar.lidar_route_nav import RouteScanSynthesizer; \
from diffuser.datasets.lidar.lidar_grid import load_or_build_grid; \
load_or_build_grid('antmaze-giant-stitch-v0', dict(n_beams=30, max_range=12.0), xy_step=1.0, n_theta=12)" \
  || { echo '!! route-nav import / grid build failed'; exit 1; }

# ---- 12/20 execution stack (identical to the validated baseline) ----
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
export LIDAR_NAV_LOOKAHEAD=${LIDAR_NAV_LOOKAHEAD:-16}
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

# ---- verify-and-hop base config (tol fixed at the validated value) ----
export LIDAR_NAV_TOPK=${LIDAR_NAV_TOPK:-5}
export LIDAR_VERIFY_HOP=1
export LIDAR_VERIFY_TOL=${LIDAR_VERIFY_TOL:-1.5}
export LIDAR_VERIFY_MAX_HOPS=${LIDAR_VERIFY_MAX_HOPS:-6}
export LIDAR_VERIFY_MIN_CONF=${LIDAR_VERIFY_MIN_CONF:-0.0}

# ---- fix toggles: LEAVE UNSET for the baseline run, SET for the fixed run ----
export LIDAR_VERIFY_REJECT_PERSIST=${LIDAR_VERIFY_REJECT_PERSIST:-1}   # 1 = old immediate-hop
export LIDAR_GOAL_REFINE=${LIDAR_GOAL_REFINE:-0}                       # 0 = off

# ---- the sweep itself: ALL 20 problems x N seeds (this job's whole point) ----
N_SEEDS=${N_SEEDS:-10}
SEEDS=$(seq -s, 0 $((N_SEEDS-1)))
LIDAR_HEATMAP_OUT=${LIDAR_HEATMAP_OUT:-rollout_results/sweep_result.json}

echo "=== SWEEP CONFIG ==="
env | grep '^LIDAR_' | sort
echo "CONFIG=$CONFIG"
echo "n_seeds=$N_SEEDS  out=$LIDAR_HEATMAP_OUT"
echo "===================="

PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=offline LIDAR_HEATMAP_OUT="$LIDAR_HEATMAP_OUT" \
python diffuser/ogb_task/ogb_maze_v1/plan_ogb_stgl_sml.py \
    --config "$CONFIG" --plan_n_ep 20 --pl_seeds "$SEEDS"

echo "=== quick peek ==="
python3 - "$LIDAR_HEATMAP_OUT" <<'PY'
import json, sys
out = sys.argv[1]
d = json.load(open(out))
n_probs = len(d['per_seed_summary'][0]['ep_is_suc'])
n_seeds = len(d['per_seed_summary'])
total_suc = sum(sum(row['ep_is_suc']) for row in d['per_seed_summary'])
total_runs = n_probs * n_seeds
print(f"{out}: {total_suc}/{total_runs} total ({n_probs} problems x {n_seeds} seeds)")
print(f"overall success rate: {100*total_suc/total_runs:.1f}%")
# per-problem success rate across seeds
per_prob = [0]*n_probs
for row in d['per_seed_summary']:
    for i, s in enumerate(row['ep_is_suc']):
        per_prob[i] += int(s)
print("per-problem success count (out of", n_seeds, "seeds):")
for i, c in enumerate(per_prob):
    print(f"  problem {i:2d}: {c}/{n_seeds}")
PY

echo "=== DONE ==="
