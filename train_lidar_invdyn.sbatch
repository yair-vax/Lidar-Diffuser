#!/bin/bash
# --- SLURM Settings ---
#SBATCH --job-name=train_lidar_invdyn
#SBATCH --output=slurm_logs/lidar_invdyn-%j.out
#SBATCH --error=slurm_logs/lidar_invdyn-%j.out
#SBATCH --partition=part-preempt
#SBATCH --qos=qos-preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude ece-ykasten1

# --- Setup Environment ---
mkdir -p slurm_logs

# Load user settings
source ~/.bashrc
source /home/projects/crml-prj10233/miniforge3/etc/profile.d/conda.sh
conda activate compdfu_ogb_release

# --- Run Code ---
# GOAL-AWARE (wide-k) LiDAR inverse dynamics (2026-07-05).
# Observation = [full 29-D ant obs, lidar(30)] (59-D); goal = a future LiDAR scan
# at k ~ Uniform[1, 60] (dataset window h64), so turns/diverging futures are
# in-distribution and the goal channel finally carries direction (see the probe
# table in LIDAR_DIAGNOSIS_AND_BASELINE.md, 2026-07-05 entries).
# Logs to its own dir: logs/.../og_antM_Gi_lidar30_full_g30d_invdyn_h64_widek_*
# AFTER TRAINING: rerun lidar_invdyn_goal_sensitivity.py on this run (acceptance:
# shuffled delta >= ~0.4, roll2/roll7 ordered, self >= ~0.2, true-MSE@k4 <= ~0.10),
# THEN 20-ep nav_true with inv_model_path -> this run dir.
CONFIG=${CONFIG:-config/ogb_invdyn/og_inv_ant/og_antM_Gi_lidar30_full_g30d_invdyn_h64_widek.py}

# stale bytecode has silently reverted code on this cluster before (see the
# repl_every no-op finding) -- clear it, and fail fast if the config is missing
find . -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null
find . -name '*.pyc' -delete 2>/dev/null
[ -f "$CONFIG" ] || { echo "!! config not found: $CONFIG (sync it first)"; exit 1; }

echo "Job started on $(hostname)  --  config: $CONFIG"
# standalone export (not just a command prefix): survives line-continuation
# mangling, and train_og_invdyn.py now also defaults to offline + falls back
# to disabled if wandb.init raises -- logging can no longer kill a run.
export WANDB_MODE=${WANDB_MODE:-offline}
echo "WANDB_MODE=$WANDB_MODE"
PYTHONDONTWRITEBYTECODE=1 \
python -u diffuser/ogb_task/og_inv_dyn/train_og_invdyn.py \
    --config "$CONFIG" \
2>&1 | tee /tmp/train_lidar_invdyn_widek.log