#!/bin/bash
# --- SLURM Settings ---
#SBATCH --job-name=train_lidar_planner
#SBATCH --output=slurm_logs/lidar_planner-%j.out
#SBATCH --error=slurm_logs/lidar_planner-%j.out
#SBATCH --partition=part-preempt
#SBATCH --qos=qos-preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00


# --- Setup Environment ---
mkdir -p slurm_logs

# Load user settings
source ~/.bashrc
source /home/projects/crml-prj10233/miniforge3/etc/profile.d/conda.sh
conda activate compdfu_ogb_release

# --- Run Code ---
# Trains the CompDiffuser planner on LiDAR observations (antmaze-giant-stitch-v0).
# EGL is forced inside the python entry script; WANDB_MODE=offline for the headless
# server. The first run builds & caches the LiDAR scans (~85 s / 1M transitions).
echo "Job started on $(hostname)"
PYTHONDONTWRITEBYTECODE=1 \
WANDB_MODE=offline \
python -u diffuser/ogb_task/ogb_maze_v1/train_ogb_stgl_sml.py \
    --config config/ogb_ant_maze/og_antM_Gi_lidar30_Cd_Stgl_PadBuf_Ft64_ts512.py \
2>&1 | tee /tmp/train_lidar_planner.log
