#!/bin/bash
##
## Train the CompDiffuser planner on LiDAR observations (antmaze-giant-stitch-v0).
## This is the LiDAR counterpart of train_ogb_stgl_sml.sh -- the agent learns to
## reconstruct sequences of LiDAR scans instead of (x, y) positions.
##
## Usage:  sh ./diffuser/ogb_task/ogb_maze_v1/train_ogb_lidar_stgl_sml.sh [GPU_IDX]
##

source ~/.bashrc
source activate compdfu_ogb_release

## optionally 'cd' to the repo folder
# cd $Your_Folder_of_This_Repo

## --------- LiDAR planner: OGBench AntMaze Giant Stitch ----------
config="config/ogb_ant_maze/og_antM_Gi_lidar30_Cd_Stgl_PadBuf_Ft64_ts512.py"

{

echo $(hostname)

PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES=${1:-0} \
python diffuser/ogb_task/ogb_maze_v1/train_ogb_stgl_sml.py --config $config

exit 0

}
