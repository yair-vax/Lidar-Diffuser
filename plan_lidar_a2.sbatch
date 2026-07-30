#!/bin/bash
# --- SLURM Settings ---
#SBATCH --job-name=lidar_a2
#SBATCH --output=slurm_logs/lidar_a2-%j.out
#SBATCH --error=slurm_logs/lidar_a2-%j.out
#SBATCH --partition=part-preempt
#SBATCH --qos=qos-preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=ece-ykasten1

# ============================================================================
# A2 -- REAL-DATASET-SCAN RIBBON along the BFS route (2026-07-07; no retrain).
#
# The goal-sensitivity probe (h12, k=4): the inv-dyn responds to real k-ahead
# scans (shuffled delta 0.25, MSE 3x) yet closed loop is plan-content-
# independent. The executed subgoals are the one input never drawn from the
# training manifold: the synthetic ribbon ray-casts at lattice poses. A2 swaps
# them for real dataset frames (LIDAR_NAV_RIBBON=data): heading-gated nearest
# frames stitched as micro-segments along the route, scans re-rendered with the
# training scanner. Executor pursuit v2 advances by pacing + proximity snap;
# do NOT rely on the policy self-stopping (self-delta only 0.11 on h12).
#
# DESIGN (2026-07-07 REVISED after the widek probe + actcfg A/B verdicts):
# the widek retrain FAILED acceptance -- probe on the h64_widek ckpt: shuffled
# delta 0.070 (target >=0.4; h12 was 0.250), self 0.023, roll2/7 0.026/0.061,
# MSE(shuffled)~=MSE(true) = rule (a) GOAL-BLIND (goal channel 1.8% of action
# var vs h12's 19.8%; k~U[1,60] diluted the informative near-k mass). Its
# closed-loop arms are MOOT and were cut. actcfg A/B: no true-vs-decoy
# separation (w3 p=0.83, w6 p=0.19) and w6 << w3 (p=0.001) -> CFG on SYNTH
# ribbons is dead, w=6 refuted. BUT CFG was only ever fed OOD synthetic goals;
# h12's 20% channel was measured on REAL pairs -> the remaining eval-only
# cells are exactly this harness:
#   (1) a2_h12      data ribbon x h12, w=1, LOOKAHEAD=3 (trained k=3)
#   (2) a2_h12_w3   data ribbon x h12, w=3 (amplify the channel on REAL input)
#
# READ (ep_min_goal_dist vs goal_tol=0.5 mj; 1 cell ~ 4 mj;
#       baseline nav_true synth x h12: best 36.0 / mean 49.2, 0/20):
#   a2_h12 << 49.2        -> ribbon OOD was the h12 blocker (quoted-A2 wins;
#                            no retrain needed -- tune + bench100).
#   a2_h12 flat, w3 helps -> channel usable only when real AND amplified;
#                            keep w=3, consider w in {2,4} sweep.
#   both ~45-50 flat      -> eval-only levers are now FULLY exhausted (ribbon,
#                            CFG, and their product) -> build + train the
#                            difference-goal inv-dyn v2 (see memory file).
#
# SYNC FIRST: ogb_stgl_sml_lidar_planner_v1.py, plan_ogb_stgl_sml.py (bytecode
# cleared below). NAV_GOAL_MODE=true isolates executor+ribbon from the
# goal-localization decoy rung (known separate issue).
# ============================================================================

mkdir -p slurm_logs rollout_results
source ~/.bashrc
source /home/projects/crml-prj10233/miniforge3/etc/profile.d/conda.sh
conda activate compdfu_ogb_release
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
echo "Job started on $(hostname)"

CONFIG=${CONFIG:-config/ogb_ant_maze/og_antM_Gi_lidar30_Cd_Stgl_PadBuf_Ft64_ts512.py}
PLANS_DIR="logs/antmaze-giant-stitch-v0/plans"
WIDEK_DIR="logs/antmaze-giant-stitch-v0/diffusion/og_antM_Gi_lidar30_full_g30d_invdyn_h64_widek"

find . -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null
find . -name '*.pyc' -delete 2>/dev/null

# ---- fail fast: edited files must compile + import on the server ----
python -m py_compile diffuser/ogb_task/ogb_maze_v1/ogb_stgl_sml_lidar_planner_v1.py \
                     diffuser/ogb_task/ogb_maze_v1/plan_ogb_stgl_sml.py \
  || { echo '!! py_compile failed -- stale sync?'; exit 1; }

# ---- shared knobs ----
export LIDAR_REPLAN=1
export LIDAR_REPL_EVERY=${LIDAR_REPL_EVERY:-250}
export LIDAR_MAX_N_REPL=15
export LIDAR_N_MAX_STEPS=${LIDAR_N_MAX_STEPS:-2500}
export LIDAR_GOAL_YAW=0.0
export LIDAR_LOC_XYSTEP=1.0
export LIDAR_LOC_NTHETA=12
export LIDAR_LOC_MOTION_SIGMA=1.0
export LIDAR_NAV_STEP_WU=0.13
export LIDAR_PICK_TYPE=first
export LIDAR_NAV_MODE=bfs
export LIDAR_NAV_GOAL_MODE=true
export LIDAR_ARRIVAL=0
export LIDAR_NAV_PURSUIT=1
## densest frame coverage for the A2 snap (1M frames vs 398k in the val split)
export LIDAR_TF_NPZ="$HOME/.ogbench/data/antmaze-giant-stitch-v0.npz"

N_EP=${N_EP:-20}

run_plan () {           # $1 = tag, $2 = n_ep
    local TAG=$1; local NEP=${2:-$N_EP}
    local STAMP; STAMP=$(mktemp); touch "$STAMP"
    echo "=== [$TAG] RIBBON=$LIDAR_NAV_RIBBON INV=${LIDAR_INV_MODEL_PATH:-h12-default} LOOKAHEAD=$LIDAR_NAV_LOOKAHEAD, $NEP ep ==="
    PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=offline \
    python diffuser/ogb_task/ogb_maze_v1/plan_ogb_stgl_sml.py \
        --config "$CONFIG" --plan_n_ep "$NEP" --pl_seeds 0
    local rc=$?
    local CFG_TAG; CFG_TAG=$(basename "$CONFIG" .py)
    local SRC; SRC=$(find "$PLANS_DIR" -path "*${CFG_TAG}*" -name 00_rollout.json -newer "$STAMP" 2>/dev/null | head -1)
    rm -f "$STAMP"
    if [ $rc -ne 0 ] || [ -z "$SRC" ]; then echo "!! [$TAG] FAILED (exit $rc, json='$SRC')"; return 1; fi
    cp "$SRC" "rollout_results/00_rollout_${TAG}.json"
    echo "saved -> rollout_results/00_rollout_${TAG}.json"
    return 0
}

peek () {
python - "$1" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); g = d.get
md = g('ep_min_goal_dist'); suc = g('ep_is_suc')
print(f"[peek] {sys.argv[1]}  ep_srate={g('ep_srate')}")
if isinstance(suc, list): print(f"   successes: {sum(bool(x) for x in suc)}/{len(suc)}")
if isinstance(md, list) and md:
    print(f"   ep_min_goal_dist: best={min(md):.1f}  mean={sum(md)/len(md):.1f}  worst={max(md):.1f} mj (1 cell~4mj)")
PY
}

# (0) SMOKE: 1 ep of the A2 ribbon on h12. Watch for '[gridnav/A2] ribbon: ..'
#     lines: expect ~hundreds of real frames, snap p50 << 0.5 wu, ~0 synth-fill.
export LIDAR_NAV_RIBBON=data
export LIDAR_NAV_LOOKAHEAD=${LIDAR_NAV_LOOKAHEAD:-3}
#unset LIDAR_INV_MODEL_PATH
if ! run_plan a2_smoke 1; then echo '!! smoke failed -- STOP'; exit 1; fi
peek rollout_results/00_rollout_a2_smoke.json

# (1) A2 x h12: the quoted-instruction arm (spacing ~= trained k=3).
if run_plan a2_h12; then peek rollout_results/00_rollout_a2_h12.json; fi

# (2) A2 x h12 x act-CFG w=3: the last untested eval-only cell. The 07-05 CFG
#     A/B amplified the goal channel while feeding it OOD SYNTH goals (flat,
#     p=0.83); here it amplifies on REAL in-distribution goals -- the regime
#     where the 20% channel was actually measured. w=6 refuted (p=0.001), so
#     w=3 only.
export LIDAR_ACT_CFG_W=3
if run_plan a2_h12_w3; then peek rollout_results/00_rollout_a2_h12_w3.json; fi
export LIDAR_ACT_CFG_W=1

## widek closed-loop arms REMOVED 2026-07-07: the h64_widek probe failed
## acceptance (goal-blind, rule (a)) -- steering arms with it cannot inform.
## To resurrect manually: LIDAR_INV_MODEL_PATH=$WIDEK_DIR LIDAR_NAV_LOOKAHEAD=30.

echo "=== DONE. Compare vs prior nav_true (synth x h12): best 36.0 / mean 49.2 mj, 0/20 ==="
