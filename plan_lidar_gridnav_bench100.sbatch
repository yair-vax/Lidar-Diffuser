#!/bin/bash
# --- SLURM Settings ---
#SBATCH --job-name=lidar_gridnav_b100
#SBATCH --output=slurm_logs/lidar_gridnav_b100-%j.out
#SBATCH --error=slurm_logs/lidar_gridnav_b100-%j.out
#SBATCH --partition=part-preempt
#SBATCH --qos=qos-preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --exclude=ece-ykasten1

# ============================================================================
# BENCH100 (REVISED 2026-07-08) -- 5 seeds x 20 ep of the 12/20 (60%) config:
#   chunk16 inv-dyn x A2 real-scan ribbon x repl_every=500 x n_max_steps=8000,
#   nav_true (BFS to the true goal cell), pursuit v2, ARRIVAL off.
#   Reference: seed 0 gave 12/20, median arrival step 4192 (00_rollout-a53f1bcd).
#
# CHANGES vs the old OP4 version (both were silent config traps):
#   - L48-class clobber FIXED: every knob is ${VAR:-default} -- nothing hardcoded.
#   - Defaults now equal the PROVEN arm (old defaults = h12 ckpt + synth ribbon
#     + 2500 steps + repl250 + NAV_GOAL_MODE=lidar + ARRIVAL=1).
#   - CONFIG ECHO block prints every LIDAR_* var -- the jsons do not record
#     LOOKAHEAD/ribbon/chunk_exec, the log is the only attribution record.
#   - Wall-clock ~5-6 h/seed (successes self-stop early, failures run 8000)
#     => 5 seeds ~25-30 h; --time=48h for headroom. Split via SEEDS= if preempted.
#
# !! VERIFY ONCE vs the 12/20 job's log: grep its '=== [...] LOOKAHEAD=' echo.
# !! If that job ran LOOKAHEAD=3 (a2-sbatch default) instead of 16, launch this
# !! bench with LIDAR_NAV_LOOKAHEAD=3 to benchmark the exact same config.
#
# To bench a different arm later (e.g. pure-LiDAR goal), override via env:
#   LIDAR_NAV_GOAL_MODE=lidar LIDAR_TF_BAND=1 LIDAR_ARRIVAL=1 sbatch ...
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
CHUNK16_DIR="logs/antmaze-giant-stitch-v0/diffusion/og_antM_Gi_lidar30_full_g30d_invdyn_h24_chunk16"

find . -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null
find . -name '*.pyc' -delete 2>/dev/null

# ---- fail fast: edited files must compile; grid cache pre-built ----
python -m py_compile diffuser/ogb_task/ogb_maze_v1/ogb_stgl_sml_lidar_planner_v1.py \
                     diffuser/ogb_task/ogb_maze_v1/ogb_stgl_sml_planner_v1.py \
                     diffuser/ogb_task/ogb_maze_v1/plan_ogb_stgl_sml.py \
  || { echo '!! py_compile failed -- stale sync?'; exit 1; }
python -c "from diffuser.datasets.lidar.lidar_route_nav import RouteScanSynthesizer; \
from diffuser.datasets.lidar.lidar_grid import load_or_build_grid; \
load_or_build_grid('antmaze-giant-stitch-v0', dict(n_beams=30, max_range=12.0), xy_step=1.0, n_theta=12)" \
  || { echo '!! route-nav import / grid build failed'; exit 1; }

# ---- the 12/20 config; every knob overridable, nothing clobbered ----
export LIDAR_REPLAN=${LIDAR_REPLAN:-1}
export LIDAR_REPL_EVERY=${LIDAR_REPL_EVERY:-500}
export LIDAR_MAX_N_REPL=${LIDAR_MAX_N_REPL:-15}          # 15 x 500 covers to step 7500
export LIDAR_N_MAX_STEPS=${LIDAR_N_MAX_STEPS:-8000}
export LIDAR_GOAL_YAW=${LIDAR_GOAL_YAW:-0.0}
export LIDAR_LOC_XYSTEP=${LIDAR_LOC_XYSTEP:-1.0}
export LIDAR_LOC_NTHETA=${LIDAR_LOC_NTHETA:-12}
export LIDAR_LOC_MOTION_SIGMA=${LIDAR_LOC_MOTION_SIGMA:-1.0}
export LIDAR_NAV_STEP_WU=${LIDAR_NAV_STEP_WU:-0.13}
export LIDAR_PICK_TYPE=${LIDAR_PICK_TYPE:-first}
export LIDAR_NAV_MODE=${LIDAR_NAV_MODE:-bfs}
export LIDAR_NAV_GOAL_MODE=${LIDAR_NAV_GOAL_MODE:-true}  # the proven arm; 'lidar' for the pure arm
export LIDAR_ARRIVAL=${LIDAR_ARRIVAL:-0}
export LIDAR_ARRIVAL_ENDS=${LIDAR_ARRIVAL_ENDS:-0}
export LIDAR_NAV_PURSUIT=${LIDAR_NAV_PURSUIT:-1}
export LIDAR_NAV_RIBBON=${LIDAR_NAV_RIBBON:-data}        # A2 real-dataset-scan ribbon
export LIDAR_NAV_LOOKAHEAD=${LIDAR_NAV_LOOKAHEAD:-16}    # chunk16 pairing; SEE HEADER WARNING
export LIDAR_INV_MODEL_PATH=${LIDAR_INV_MODEL_PATH:-$CHUNK16_DIR}
export LIDAR_INV_EPOCH=${LIDAR_INV_EPOCH:-1600000}
## densest frame coverage for the A2 snap (1M frames vs 398k in the val split)
export LIDAR_TF_NPZ=${LIDAR_TF_NPZ:-"$HOME/.ogbench/data/antmaze-giant-stitch-v0.npz"}
# passthrough (code defaults if unset): LIDAR_ACT_CHUNK_EXEC (m/2=8),
# LIDAR_NAV_A2_SEG (12), LIDAR_NAV_A2_SNAP_WU (2.0), pursuit v2 knobs.

# ---- CONFIG ECHO: the jsons don't record these; this log line is the record ----
echo "=== BENCH CONFIG ==="
env | grep '^LIDAR_' | sort
echo "CONFIG=$CONFIG"
echo "===================="

N_EP=${N_EP:-20}
SEEDS=${SEEDS:-"0 1 2 3 4"}

for SD in $SEEDS; do
    STAMP=$(mktemp); touch "$STAMP"
    echo "=== [bench seed $SD] NAV=$LIDAR_NAV_MODE GOAL=$LIDAR_NAV_GOAL_MODE RIBBON=$LIDAR_NAV_RIBBON LOOKAHEAD=$LIDAR_NAV_LOOKAHEAD INV=$LIDAR_INV_MODEL_PATH STEPS=$LIDAR_N_MAX_STEPS REPL=$LIDAR_REPL_EVERY, $N_EP ep ==="
    PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=offline \
    python diffuser/ogb_task/ogb_maze_v1/plan_ogb_stgl_sml.py \
        --config "$CONFIG" --plan_n_ep "$N_EP" --pl_seeds "$SD"
    CFG_TAG=$(basename "$CONFIG" .py)
    SRC=$(find "$PLANS_DIR" -path "*${CFG_TAG}*" -name 00_rollout.json -newer "$STAMP" 2>/dev/null | head -1)
    rm -f "$STAMP"
    if [ -n "$SRC" ]; then
        cp "$SRC" "rollout_results/00_rollout_bench_s${SD}.json"
        echo "saved -> rollout_results/00_rollout_bench_s${SD}.json"
        # sanity: fail loudly if the json does not reflect the requested budget
        python - "rollout_results/00_rollout_bench_s${SD}.json" <<'PY'
import json, os, sys
d = json.load(open(sys.argv[1]))
want = int(os.environ.get('LIDAR_N_MAX_STEPS', '8000'))
got = d.get('repl_ada_dist_cfg', {}).get('n_max_steps'), d.get('max_episode_steps')
print(f"  [check] n_max_steps/max_episode_steps in json: {got} (want {want})")
if any(g != want for g in got): print("  !! BUDGET MISMATCH -- a knob was clobbered somewhere; do not trust this seed")
PY
    else
        echo "!! [bench seed $SD] no rollout json found"
    fi
done

echo "=== POOLED SUMMARY ==="
python - <<'PY'
import glob, json, math
suc, md, n, arr = 0, [], 0, []
for p in sorted(glob.glob('rollout_results/00_rollout_bench_s*.json')):
    d = json.load(open(p))
    s = [bool(x) for x in d.get('ep_is_suc', [])]
    m = d.get('ep_min_goal_dist', []); a = d.get('ep_argmin_step', [])
    suc += sum(s); n += len(s); md += list(m)
    arr += [a[i] for i in range(len(s)) if s[i]]
    print(f"  {p}: {sum(s)}/{len(s)}  srate={d.get('ep_srate')}  epoch={d.get('epoch_diffusion')}")
if n:
    ph = suc / n; z = 1.96
    den = 1 + z*z/n
    ctr = (ph + z*z/(2*n)) / den
    hw  = z * math.sqrt(ph*(1-ph)/n + z*z/(4*n*n)) / den
    print(f"POOLED: {suc}/{n} = {100*ph:.1f}%  Wilson95 [{100*(ctr-hw):.1f}%, {100*(ctr+hw):.1f}%]")
    print(f"  min_goal_dist mean={sum(md)/max(len(md),1):.1f} best={min(md):.1f} mj (1 cell~4mj; tol 0.5)")
    if arr:
        arr.sort()
        print(f"  arrival steps: median={arr[len(arr)//2]}  " +
              "  ".join(f"suc@{t}={sum(x<=t for x in arr)}/{n}" for t in (2500, 4000, 5000, 6000, 7000, 8000)))
PY
