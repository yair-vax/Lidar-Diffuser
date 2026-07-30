#!/bin/bash
##
## Train the inverse-dynamics model for the LiDAR giant antmaze planner.
## Observation = [x, y, cos(yaw), sin(yaw), lidar...]; goal = next LiDAR scan.
## These checkpoints are consumed by the giant planner to turn generated LiDAR
## plans into executable actions.
##
## Usage:  sh ./diffuser/ogb_task/og_inv_dyn/train_og_lidar_invdyn.sh [GPU_IDX]
##

source ~/.bashrc
source activate compdfu_ogb_release

## optionally 'cd' to the repo folder
# cd $Folder

config="config/ogb_invdyn/og_inv_ant/og_antM_Gi_lidar30_xyo_g30d_invdyn_h12.py"

{

PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES=${1:-0} \
python diffuser/ogb_task/og_inv_dyn/train_og_invdyn.py --config $config

exit 0

}
