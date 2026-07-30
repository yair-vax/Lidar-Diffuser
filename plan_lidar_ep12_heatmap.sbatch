#!/bin/bash
# --- SLURM Settings ---
#SBATCH --job-name=lidar_ep12hm
#SBATCH --output=slurm_logs/lidar_ep12hm-%j.out
#SBATCH --error=slurm_logs/lidar_ep12hm-%j.out
#SBATCH --partition=part-preempt
#SBATCH --qos=qos-preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --exclude=ece-ykasten1

# ============================================================================
# REPEAT EPISODE 12 x100 -- FINAL-POSITION HEATMAP (2026-07-15).
#
# Episode 12 (eval problem index 12, seed pl_seed=0) was one of two episodes
# that actually succeeded under the validated single-scan verify-hop config
# (tol=1.5, no walking-distance prior, no sequence match -- the exact config
# reverted back to after the walkdist/gate/sequence-match experiments were
# undone): min_goal_dist=0.04, essentially exact.
#
# This job repeats ONLY that one start/goal pair (LIDAR_EP_ST_IDX=12,
# --plan_n_ep 1) 100 times with different pl_seeds (0..99), reusing the SAME
# loaded model/planner across all 100 runs (new multi-seed branch in
# plan_ogb_stgl_sml.py main(), previously a NotImplementedError stub) instead
# of paying the model-load cost 100 times. Captures the ACTUAL final (x, y)
# position (self.env.get_xy(), same call the per-step goal-distance logging
# already used) after each run -> rollout_results/ep12_100x_heatmap.json.
#
# Everything else matches plan_lidar_verifyhop_probe.sbatch exactly (the
# validated 2/20 config) so this is a clean repeatability probe, not a new
# variant: does episode 12's start/goal pair reliably converge to the goal
# across sampling noise, or was the earlier single success a lucky seed?
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

# ---- 12/20 execution stack (identical to the validated 2/20 config) ----
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

# ---- verify-and-hop: the validated 2/20 config ----
export LIDAR_NAV_TOPK=${LIDAR_NAV_TOPK:-5}
export LIDAR_VERIFY_HOP=1
export LIDAR_VERIFY_TOL=${LIDAR_VERIFY_TOL:-1.5}
export LIDAR_VERIFY_MAX_HOPS=${LIDAR_VERIFY_MAX_HOPS:-6}
export LIDAR_VERIFY_MIN_CONF=${LIDAR_VERIFY_MIN_CONF:-0.0}

# ---- NEW: repeat ONE episode across many seeds (this job's whole point) ----
export LIDAR_EP_ST_IDX=${LIDAR_EP_ST_IDX:-12}
export LIDAR_HEATMAP_OUT=${LIDAR_HEATMAP_OUT:-rollout_results/ep12_100x_heatmap.json}
N_REPEATS=${N_REPEATS:-100}
SEEDS=$(seq -s, 0 $((N_REPEATS-1)))

echo "=== PROBE CONFIG ==="
env | grep '^LIDAR_' | sort
echo "CONFIG=$CONFIG"
echo "ep_st_idx=$LIDAR_EP_ST_IDX  n_repeats=$N_REPEATS"
echo "===================="

PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=offline \
python diffuser/ogb_task/ogb_maze_v1/plan_ogb_stgl_sml.py \
    --config "$CONFIG" --plan_n_ep 1 --pl_seeds "$SEEDS"

echo "=== quick peek ==="
python3 - <<PY
import json
d = json.load(open("$LIDAR_HEATMAP_OUT"))
s = [row['is_suc'] for row in d['per_seed_summary']]
print(f"succeeded: {sum(s)}/{len(s)}")
print(f"true_goal_xy: {d['true_goal_xy']}")
PY

echo "=== DONE ==="
