:construction: Work in progress...

# Quick steps to get it up and running:


## Clone repo

Clone the repo with submodules:
```bash
git clone --recurse-submodules git@github.com:art-e-fact/arena-shape-sorting.git
```
or, if already cloned, init the submodules:
```bash
git submodule update --init --recursive
```

## Install

### With Docker 

This will build the container and start an interactive shell.
```bash
docker/run_docker.sh
```

### With Python venv

Requires [uv](https://docs.astral.sh/uv/) and a machine that can run Isaac Sim / Isaac Lab.
Source once per shell session (creates/syncs Arena's venv on first use, then activates it):

```bash
source ./setup.sh
# optional: source ./setup.sh --force   # re-run uv sync
# optional: source ./setup.sh --wheel   # Isaac Lab from wheel instead of source
```

> TODO: We can switch to simple `uv run ...` once IsaacLab-Arena will be released as a Python package 

## Start demonstrations

Segmented recording (recommended): plan a motion without recording, then replay or
smooth-execute it from a checkpoint. Only the execute phase is written to HDF5.

Keys while running: `I` retry, `U` undo segment, `P` replay, `O` smooth, `Backspace` abort, `R` reset.

SE3 keyboard / gamepad:
```bash
python -m shape_sorting.run_record_demos_segmented \
  --viz kit \
  --device cpu \
  --dataset_file ./so101_shape_sorting.hdf5 \
  --num_demos 10 \
  --num_success_steps 2 \
  shape_sorting_test \
  --embodiment so101_ik \
  --teleop_device keyboard  # or --teleop_device gamepad
```

Joint-space gamepad (absolute joints — recommended for SO-101):
```bash
python -m shape_sorting.run_record_demos_segmented \
  --viz kit \
  --device cpu \
  --dataset_file ./so101_shape_sorting.hdf5 \
  --num_demos 10 \
  --num_success_steps 2 \
  shape_sorting_test \
  --embodiment so101_abs_joint \
  --teleop_device gamepad
```

SO-101 leader arm (absolute joints — prefer `P` replay; `O` smooth also works well here):
```bash
python -m shape_sorting.run_record_demos_segmented \
  --viz kit \
  --device cpu \
  --dataset_file ./so101_shape_sorting.hdf5 \
  --num_demos 10 \
  --num_success_steps 2 \
  shape_sorting_test \
  --embodiment so101_abs_joint \
  --teleop_device so101_leader \
  --leader_port /dev/ttyACM0
```

Notes:
 - Pass `--forms cube cylinder triangle hexagon star cross` to change the shapes used
 - Optional `--smooth_steps N` controls how many steps `O` interpolates (default 30)