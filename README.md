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

## Build Docker 

This will build the container and start an interactive shell. Until the release IsaacLab-Arena on pypi, Docker is the recommended way to install it.
```bash
docker/run_docker.sh
```

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