#!/bin/bash
# --- SLURM Settings ---
#SBATCH --job-name=cooking
#SBATCH --output=slurm_logs/cooking-%j.out
#SBATCH --error=slurm_logs/cooking-%j.out
#SBATCH --partition=part-preempt
#SBATCH --qos=qos-preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --exclude=ece-ykasten1

# --- Setup Environment ---
mkdir -p slurm_logs rollout_results


# Load user settings
source ~/.bashrc
source /home/projects/crml-prj10233/miniforge3/etc/profile.d/conda.sh
conda activate compdfu_ogb_release

# --- Run Code ---
echo "Job started on $(hostname)"
export MUJOCO_GL=egl; export PYOPENGL_PLATFORM=egl
 

CONFIG="config/ogb_ant_maze/og_antM_Gi_lidar30_Cd_Stgl_PadBuf_Ft64_ts512.py"
PLANS_DIR="logs/antmaze-giant-stitch-v0/plans"
 
find . -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null
find . -name '*.pyc' -delete 2>/dev/null
 
# fixed knobs == the K=4 baseline (synthetic goal, periodic replan, cond_w 2). NO tf_band.
export LIDAR_REPLAN=1
export LIDAR_REPL_EVERY=0
export LIDAR_COND_W=2.0
export LIDAR_GOAL_YAW=0.0
 
N_EP=${N_EP:-20}
 
run_plan () {           # $1 = tag
    local TAG=$1
    local STAMP; STAMP=$(mktemp); touch "$STAMP"
    echo "=== [$TAG] PICK=$LIDAR_PICK_TYPE TOP_N=$LIDAR_TOP_N BSIZE=$LIDAR_BSIZE, $N_EP ep ==="
    PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=offline \
    python diffuser/ogb_task/ogb_maze_v1/plan_ogb_stgl_sml.py \
        --config "$CONFIG" --plan_n_ep "$N_EP" --pl_seeds 0
    local rc=$?
    local SRC; SRC=$(find "$PLANS_DIR" -name 00_rollout.json -newer "$STAMP" 2>/dev/null | head -1)
    rm -f "$STAMP"
    if [ $rc -ne 0 ] || [ -z "$SRC" ]; then echo "!! [$TAG] FAILED (exit $rc, json='$SRC')"; return 1; fi
    cp "$SRC" "rollout_results/00_rollout_${TAG}.json"
    echo "saved -> rollout_results/00_rollout_${TAG}.json"
    return 0
}
 
peek () {               # $1 = json path
python - "$1" <<'PY'
import json, sys
d=json.load(open(sys.argv[1])); g=d.get
md=g('ep_min_goal_dist')
print(f"[peek] {sys.argv[1]}  srate={g('ep_srate')}")
if isinstance(md,list) and md:
    print(f"   ep_min_goal_dist: best={min(md):.1f}  mean={sum(md)/len(md):.1f}  worst={max(md):.1f} mj  (1 cell~4mj)")
    print(f"     per-ep: {[round(x,1) for x in md]}")
PY
}
 
# (1) BASELINE -- consistency selection, current pool (== today's K=4 run).
export LIDAR_PICK_TYPE=first; export LIDAR_TOP_N=5;  export LIDAR_BSIZE=40
if run_plan first_t5; then peek rollout_results/00_rollout_first_t5.json; fi
 
# (2) GOAL selection, SAME pool -- isolates "goal-aware vs consistency" on identical candidates.
export LIDAR_PICK_TYPE=goal;  export LIDAR_TOP_N=5;  export LIDAR_BSIZE=40
if run_plan goal_t5; then peek rollout_results/00_rollout_goal_t5.json; fi
 
# (3) CONTROL: consistency selection, BIGGER pool -- isolates POOL SIZE from selection.
#     Compare (3) vs (4): if first_t20 ~= goal_t20, the gain is just more sampling, not
#     smarter picking; if goal_t20 < first_t20, goal selection earns its keep at scale.
export LIDAR_PICK_TYPE=first; export LIDAR_TOP_N=20; export LIDAR_BSIZE=64
if run_plan first_t20; then peek rollout_results/00_rollout_first_t20.json; fi
 
# (4) GOAL selection, BIGGER pool -- more candidates sampled AND re-ranked.
export LIDAR_PICK_TYPE=goal;  export LIDAR_TOP_N=20; export LIDAR_BSIZE=64
if run_plan goal_t20; then peek rollout_results/00_rollout_goal_t20.json; fi
 
echo "=== DONE. ep_min_goal_dist: (1)first_t5 (2)goal_t5 (3)first_t20 (4)goal_t20."
echo "    pool size: (1)vs(3).  selection at scale: (3)vs(4).  selection small: (1)vs(2). (lower=better) ==="