:construction: Work in progress...

# IsaacLab-Arena shape-sorting environment and SO-101 embodiment

## Set up workspace

### Clone repo

Clone the repo with submodules:
```bash
git clone --recurse-submodules git@github.com:art-e-fact/arena-shape-sorting.git
```
or, if already cloned, init the submodules:
```bash
git submodule update --init --recursive
```

### With Docker

Build the container and start an interactive shell:
```bash
./docker/run_docker.sh
```

### With Python venv

Uses Arena's [native uv setup](https://isaac-sim.github.io/IsaacLab-Arena/release/0.3.0-prerelease/pages/quickstart/installation.html).

Source once per shell session (creates/syncs Arena's venv on first use, then activates it):

```bash
source ./setup.sh
# optional: source ./setup.sh --force   # re-run uv sync
# optional: source ./setup.sh --wheel   # Isaac Lab from wheel instead of source
```

The virtual environment will be located under `submodules/IsaacLab-Arena/.venv`.

> TODO: We can switch to a normal `pyproject.toml` and `uv run ...` once IsaacLab-Arena is released as a Python package.

## Run the environment

Smoke-test the installation by running the environment with a zero-action policy:
```bash
python submodules/IsaacLab-Arena/isaaclab_arena/evaluation/policy_runner.py \
  --viz kit \
  --policy_type zero_action \
  --num_steps 50 \
  --external_environment_class_path shape_sorting.shape_sorting_env:ShapeSortingEnvironment \
  shape_sorting_test \
  --forms cube cylinder hexagon star
```
The viewer should show the environment with the default embodiment.
<img width="2488" height="1378" alt="shape-sorting-env-kit-franka" src="https://github.com/user-attachments/assets/adc9c370-7958-4514-8bce-4b68b8a02dad" />





### Environment options (`shape_sorting_test`)

These flags go after the `shape_sorting_test` subcommand (same for `policy_runner.py`, `record_demos.py`, and the segmented recorder):

| Flag | Default | Description |
|------|---------|-------------|
| `--embodiment` | `droid_rel_joint_pos` | Robot embodiment registry name (`so101_ik`, `so101_abs_joint`, …) |
| `--teleop_device` | none | Teleop device (`keyboard`, `gamepad`, `spacemouse`, `so101_leader`, …) |
| `--leader_port` | `/dev/ttyACM0` | Serial port for `so101_leader` |
| `--leader_id` | `leader` | Leader arm id |
| `--leader_recalibrate` | off | Recalibrate the leader arm on start |
| `--hdr` | none | HDR map name (e.g. `home_office_robolab`) |
| `--light_intensity` | `500.0` | Scene light intensity |
| `--additional_table_objects` | none | Extra asset registry names to place on the table |
| `--forms` | `cube cylinder hexagon` | Shape silhouettes; choices: `cube`, `cylinder`, `triangle`, `hexagon`, `star`, `cross` |
| `--piece_size` | `0.03` | Equal-area reference square side length (m) |
| `--piece_height` | `0.03` | Piece extrusion height (m) |
| `--box_height` | `0.04` | Sorting box height (m) |
| `--clearance` | `0.003` | Hole clearance around each piece (m) |
| `--edge_chamfer` | `0.001` | Piece top/bottom edge chamfer (m) |
| `--hole_chamfer` | `0.001` | Hole rim lead-in chamfer (m) |

`--enable_cameras` is a shared Arena flag (pass it before `shape_sorting_test`), not an env-subcommand option.

## Teleoperation data collection



https://github.com/user-attachments/assets/6e2105bf-f46c-4b04-8061-7cf49cbc7e35



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
  --teleop_device keyboard
```

### Other tested teleop options for the SO-101 embodiment

*See the [SO-101 embodiment](arena_so101/README.md#joint-space-gamepad-layout-so101_abs_joint--gamepad) for more detail.*

SE(3) differential gamepad:
```bash
  ...
  --embodiment so101_ik \
  --teleop_device gamepad
```

Joint-space gamepad (absolute joints — recommended for SO-101):

```bash
  ...
  --embodiment so101_abs_joint \
  --teleop_device gamepad
```

Teleop with the SO-101 leader arm:
```bash
  ...
  --embodiment so101_abs_joint \
  --teleop_device so101_leader \
  --leader_port /dev/ttyACM0
```

### Experimental: Segmented recording

Executing smooth, error-free demonstrations can be challenging. This script records segment-by-segment.

Features:
 - Undo parts of the demo
 - Replace teleoperated motion with a smooth transition from the start to the end position

The CLI will guide you through the recording process. Optional `--smooth_steps N` controls how many steps `O` interpolates (default 30).

```bash
python -m shape_sorting.run_record_demos_segmented \
  --viz kit \
  --device cpu \
  --dataset_file ./so101_shape_sorting.hdf5 \
  --num_demos 10 \
  --num_success_steps 2 \
  --external_environment_class_path shape_sorting.shape_sorting_env:ShapeSortingEnvironment \
  shape_sorting_test \
  --embodiment so101_ik \
  --teleop_device keyboard
```
