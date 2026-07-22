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

## Start demonstraions

Start recording teleoperated demonstraions with SE3 controllers. (Due to the limited degrees of freedoms the SO101 have, controlling the the end-effector target pose is not effective but good enough for testing the setup)
```bash
python submodules/IsaacLab-Arena/isaaclab_arena/scripts/imitation_learning/record_demos.py \
  --viz kit \
  --device cpu \
  --dataset_file ./so101_shape_sorting.hdf5 \
  --num_demos 10 \
  --num_success_steps 2 \
  --external_environment_class_path shape_sorting.shape_sorting_env:ShapeSortingEnvironment \
  shape_sorting_test \
  --embodiment so101_ik \
  --teleop_device keyboard  # or --teleop_device gamepad
```

Start recording teleoperated demonstraions with the SO101 leader arm.
```bash
python submodules/IsaacLab-Arena/isaaclab_arena/scripts/imitation_learning/record_demos.py \
  --viz kit \
  --device cpu \
  --dataset_file ./so101_shape_sorting.hdf5 \
  --num_demos 10 \
  --num_success_steps 2 \
  --external_environment_class_path shape_sorting.shape_sorting_env:ShapeSortingEnvironment \
  shape_sorting_test \
  --embodiment so101_abs_joint \
  --teleop_device so101_leader \
  --leader_port /dev/ttyACM0
```

Notes:
 - Pass `--forms cube cylinder triangle hexagon star cross` to change the shapes used