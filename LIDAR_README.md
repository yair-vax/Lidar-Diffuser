# LiDAR-based CompDiffuser (AntMaze Giant)

This is a transformation of the CompDiffuser codebase so that the planner is
trained on **simulated LiDAR observations instead of the agent's `(x, y)`
position**. The model never sees the position label ("the *y*"): it learns to
denoise / reconstruct *sequences of LiDAR scans*, conditioned on a start scan and
a goal scan. The work targets the **`antmaze-giant-stitch-v0`** ("giant") model.

It implements the project specification:

| Spec stage | Where it lives |
| --- | --- |
| 1. Repo adaptation (LiDAR support) | new modules + configs below, existing classes reused |
| 2. LiDAR simulation (position + orientation → ray distances) | `diffuser/datasets/lidar/lidar_sim.py` |
| 3. Dataset management (exact copy in LiDAR format) | `diffuser/datasets/ogb_dset/ogb_lidar_utils.py` (auto-cache) + `diffuser/datasets/lidar/gen_lidar_dataset.py` (explicit copy) |
| 4. Train the LiDAR planner from scratch (~30-dim obs) | `config/ogb_ant_maze/og_antM_Gi_lidar30_Cd_Stgl_PadBuf_Ft64_ts512.py` |
| 5. Train the inverse-dynamics model (x, y, orientation, LiDAR → action) | `config/ogb_invdyn/og_inv_ant/og_antM_Gi_lidar30_xyo_g30d_invdyn_h12.py` |
| 6. Evaluation: test-set position histograms | `diffuser/pl_eval/lidar_eval/lidar_pos_histogram.py` |

Everything is built on top of the existing, tested training/diffusion machinery —
the observation space is the only thing that changes, and the U-Net adapts to it
automatically (its transition dim is read from `dataset.observation_dim`).

---

## 1. The LiDAR observation

The agent's `(x, y)` position is replaced by a **30-beam, 360°, body-frame LiDAR
scan**: a vector of ray distances to the nearest maze wall.

* **Input** (`obs_layout='lidar'`): the 30 ray distances (≈ the "~30 dimensions"
  of the spec). This is what the diffusion planner sees.
* **Orientation dependence** ("car-like"): the beams are cast relative to the
  agent's heading `yaw`, which is recovered from the torso quaternion
  (`obs[3:7]`, `w,x,y,z`). So the scan rotates with the agent.
* **Geometry**: ray-marching against the maze occupancy grid, using the exact
  OGBench world↔cell convention (`maze_unit=4`, `offset=4`, cell `(i,j)` centered
  at `x=j*4-4, y=i*4-4`). Verified: 100% of dataset positions fall in free cells,
  and controlled-case distances are exact.

Defaults (in the configs, tweak freely):

```python
n_beams   = 30        # observation dimension
max_range = 12.0      # world units (3 maze cells); beams clamp here if no hit
fov       = 2*pi      # full 360° scan
step      = 0.1       # ray-march resolution (world units)
```

The inverse-dynamics model uses a richer layout, `obs_layout='xyo_lidar'`:
`[x, y, cos(yaw), sin(yaw), lidar_0 ... lidar_29]` (34 dims) — i.e. exactly the
spec's "position, orientation and LiDAR readings".

---

## 2. New / modified files

**New**

```
diffuser/datasets/lidar/__init__.py
diffuser/datasets/lidar/lidar_sim.py            # LiDAR simulator (numpy only)
diffuser/datasets/lidar/gen_lidar_dataset.py    # write an explicit LiDAR-format copy
diffuser/datasets/ogb_dset/ogb_lidar_utils.py   # dataset<->LiDAR glue + disk cache
diffuser/datasets/ogb_dset/ogb_lidar_dataset_v1.py  # named LiDAR dataset classes
diffuser/ogb_task/ogb_maze_v1/ogb_stgl_sml_lidar_training_v1.py  # LiDAR-aware trainer
diffuser/ogb_task/ogb_maze_v1/train_ogb_lidar_stgl_sml.sh
diffuser/ogb_task/og_inv_dyn/train_og_lidar_invdyn.sh
diffuser/pl_eval/lidar_eval/lidar_pos_histogram.py  # stage-6 evaluation
config/ogb_ant_maze/og_antM_Gi_lidar30_Cd_Stgl_PadBuf_Ft64_ts512.py
config/ogb_invdyn/og_inv_ant/og_antM_Gi_lidar30_xyo_g30d_invdyn_h12.py
```

**Modified (minimal, backward compatible)**

```
diffuser/datasets/ogb_dset/ogb_utils.py   # ogb_sequence_dataset: LiDAR branch when
                                          #   dataset_config['lidar_config'] is set
diffuser/datasets/ogb_dset/__init__.py    # export the LiDAR dataset classes
diffuser/ogb_task/ogb_maze_v1/__init__.py # export the LiDAR trainer
```

If `lidar_config` is absent the original `(x, y)` behavior is completely
unchanged, so all existing configs/experiments keep working.

---

## 3. Train the LiDAR planner (giant, from scratch)

On a headless SLURM server (see [§7](#7-running-on-a-headless-gpu-server-slurm)):

```bash
sbatch train_lidar_planner.sbatch
```

Interactively / locally:

```bash
sh ./diffuser/ogb_task/ogb_maze_v1/train_ogb_lidar_stgl_sml.sh 0
# or directly:
python diffuser/ogb_task/ogb_maze_v1/train_ogb_stgl_sml.py \
    --config config/ogb_ant_maze/og_antM_Gi_lidar30_Cd_Stgl_PadBuf_Ft64_ts512.py
```

On the first run the LiDAR scans for the whole dataset are computed once
(~85 s / 1M transitions) and cached to `data/ogb_maze/lidar_cache/`; subsequent
runs load the cache instantly. Training-time visualizations are saved as LiDAR
`(time × beam)` heatmaps (reference vs. generated) instead of maze plots.

> **Debugging on the small bundled subset:** uncomment the `dset_h5path` line in
> the config to use `antmaze-giant-stitch-v0-luotest.npz` for fast data loading.
>
> **About "~200 steps" (spec stage 4):** the planner is trained on trajectory
> chunks of horizon 160 over the OGBench 200-step episodes; the full training run
> default is `n_train_steps = 2e6` (unchanged from the original giant model). For
> a quick smoke test lower `n_train_steps` / `n_steps_per_epoch` in the config.

---

## 4. Train the LiDAR inverse-dynamics model

On a headless SLURM server:

```bash
sbatch train_lidar_invdyn.sbatch
```

Interactively / locally:

```bash
sh ./diffuser/ogb_task/og_inv_dyn/train_og_lidar_invdyn.sh 0
# or directly:
python diffuser/ogb_task/og_inv_dyn/train_og_invdyn.py \
    --config config/ogb_invdyn/og_inv_ant/og_antM_Gi_lidar30_xyo_g30d_invdyn_h12.py
```

The model maps `[x, y, cos(yaw), sin(yaw), lidar]_t` together with the **next
LiDAR scan** (the goal the planner produces) to the 8-D action. Its checkpoints
are what the giant planner uses to execute generated LiDAR plans (spec stage 5).

---

## 5. (Optional) Materialize an explicit LiDAR dataset copy

Training does *not* require this (the loader caches LiDAR on the fly), but it is
handy for inspection / portability and to feed the evaluation script:

```bash
# from the bundled subset (no OGBench needed):
python -m diffuser.datasets.lidar.gen_lidar_dataset \
    --npz data/ogb_maze/antmaze-giant-stitch-v0-luotest.npz \
    --env antmaze-giant-stitch-v0 \
    --out data/ogb_maze/lidar/antmaze-giant-stitch-v0-luotest_lidar.npz

# from the full OGBench dataset (in the compdfu_ogb_release conda env):
python -m diffuser.datasets.lidar.gen_lidar_dataset \
    --env antmaze-giant-stitch-v0 \
    --out data/ogb_maze/lidar/antmaze-giant-stitch-v0_lidar.npz
```

---

## 6. Evaluation: test-set position histograms (spec stage 6)

```bash
python diffuser/pl_eval/lidar_eval/lidar_pos_histogram.py \
    --npz data/ogb_maze/antmaze-giant-stitch-v0-luotest.npz \
    --env antmaze-giant-stitch-v0 \
    --out logs/lidar_eval/giant_luotest
```

Produces (overlaid on the maze):

1. **`test_position_histogram.png`** — where the held-out trajectories go.
2. **`lidar_ambiguity_map.png`** — per-cell LiDAR distinctiveness; structurally
   ambiguous locations (open rooms, symmetric corridors) are intrinsically harder
   to reconstruct from LiDAR — the structure↔difficulty correlation the spec asks
   for.
3. **`reconstruction_error_map.png`** + **`ambiguity_vs_error_scatter.png`** —
   *only if* you pass `--recon_npz`, a model-output file with `true_xy` plus
   either (`pred_lidar`, `true_lidar`) or `pred_xy`. The per-cell reconstruction
   error is mapped onto the maze and correlated with the ambiguity map.

The script is numpy + matplotlib only (no torch/OGBench needed) and accepts both
the raw OGBench `.npz` and a LiDAR-format `.npz` from step 5.

---

## 7. Running on a headless GPU server (SLURM)

Two ready-to-use SLURM scripts are provided in the repo root, mirroring the
cluster settings of `train.sbatch` (partition `part-preempt`, conda env
`compdfu_ogb_release`):

```bash
sbatch train_lidar_planner.sbatch    # LiDAR planner  (giant)
sbatch train_lidar_invdyn.sbatch     # LiDAR inverse dynamics
```

Each script sets `WANDB_MODE=offline` and runs the python entry point directly,
e.g. the planner one runs:

```bash
PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=offline \
python -u diffuser/ogb_task/ogb_maze_v1/train_ogb_stgl_sml.py \
    --config config/ogb_ant_maze/og_antM_Gi_lidar30_Cd_Stgl_PadBuf_Ft64_ts512.py
```

To adapt your existing `train.sbatch`, just swap the `--config` (and the python
entry script for the planner) for the LiDAR configs above.

**Headless rendering (EGL).** The official repo defaulted the MuJoCo/OpenGL
backend to `osmesa`, which throws `glGetError` on these NVIDIA servers. The two
python entry scripts now force EGL at the very top (before any mujoco import):

```python
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['MUJOCO_GL'] = 'egl'
```

This is applied in `train_ogb_stgl_sml.py`, `train_og_invdyn.py` (used by the
LiDAR planner and LiDAR inverse-dynamics training respectively) and in
`gen_lidar_dataset.py`. The diffusion trainer also wraps the loss in a
`bfloat16` autocast for speed/memory — the LiDAR planner trainer inherits this
automatically. So the LiDAR training paths get exactly the same EGL / wandb /
autocast fixes you applied to the original training scripts; no LiDAR-specific
file sets the backend itself (env vars must be set in the process entry point,
which is what runs the LiDAR configs).

> If a particular node lacks EGL and you want osmesa instead, set
> `MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa` in the sbatch before the python
> call (the `gen_lidar_dataset.py` defaults are overridable; the train scripts
> set EGL unconditionally — edit those two lines if you need osmesa there).


## 8. Do I need the previously-trained (x, y / 29-D) models?

**No.** The LiDAR planner and LiDAR inverse-dynamics model are trained **from
scratch** (spec stage 4). The observation space is different — 30-D LiDAR for the
planner, 34-D `[x, y, cos, sin, lidar]` for the inverse dynamics — so the old
`o2d` / `o29d` checkpoints are dimensionally incompatible and are **not loaded or
needed**. The only input you need from the original setup is the **OGBench
dataset** itself (`antmaze-giant-stitch-v0`), from which the LiDAR scans are
generated.

What *is* reused is the *pattern*: spec stage 5's "take the checkpoints of the
inverse for the giant model" means the **newly trained LiDAR inverse-dynamics
checkpoints** are what the LiDAR giant planner would use to convert generated
LiDAR plans into actions at rollout time — not any pre-existing weights.


## 9. Closed-loop rollout in MuJoCo (remaining integration)

Training (planner + inverse dynamics) and the histogram evaluation are fully
wired and runnable. The original closed-loop rollout
(`diffuser/ogb_task/ogb_maze_v1/plan_ogb_stgl_sml.py`) still assumes an `(x, y)`
state and would need one extra piece to run *with the LiDAR planner*: an online
LiDAR wrapper that, at each environment step, turns the live MuJoCo state into a
LiDAR scan (using `LidarScanner` / `make_scanner_for_env`) and conditions the
planner on the start/goal LiDAR. The simulator already exposes exactly the call
needed — `scanner.scan(x, y, yaw)` — so this is a localized addition; it is left
out here because stage 6 specifies *histogram-based* evaluation rather than the
MuJoCo rollout.
```python
from diffuser.datasets.lidar import make_scanner_for_env
scanner = make_scanner_for_env('antmaze-giant-stitch-v0', dict(n_beams=30, max_range=12.0))
lidar_t = scanner.scan(x, y, yaw)   # 30-vector to condition the planner online
```
