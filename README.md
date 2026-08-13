# IsaacLab-Arena shape-sorting environment and SO-101 embodiment

Example workspace for solving a shape-sorting task with an SO-101 arm in a procedurally generated environment.

Main features:
 - Procedurally generated environment
 - SO-101 arm embodiment
 - Scripted synthetic dataset generation for imitation learning
 - Teleoperation and data collection
 - Training and evaluation with LeRobot

Contents:
 - [Setup workspace](#setup-workspace)
    - [Clone repo](#clone-repo)
    - [With Docker](#with-docker)
    - [With Python venv (uv)](#with-python-venv)
 - [Run the environment](#run-the-environment)
    - [Smoke test](#smoke-test)
    - [cuRobo SO-101 reach smoke test](#curobo-so-101-reach-smoke-test)
    - [Environment options](#environment-options)
 - [Building the dataset](#building-the-dataset)
    - [Scripted synthetic dataset generation](#scripted-synthetic-dataset-generation)
    - [Teleoperation data collection](#teleoperation-data-collection)
 - [Training and evaluation with LeRobot](#training-and-evaluation-with-lerobot)
    - [Train an ACT policy on the shape-sorting dataset](#train-an-act-policy-on-the-shape-sorting-dataset)
    - [Evaluate with the LeRobot CLI](#evaluate-with-the-lerobot-cli)
    - [Run the evaluation with Artefacts](#run-the-evaluation-with-artefacts)

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

### With Python venv (uv)

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

### cuRobo SO-101 reach smoke test

Plans once to a fixed EE pose with cuRobo, then plays absolute joint waypoints
(``so101_abs_joint``). Requires a generated ``so101.yml``
(``python -m arena_so101.generate_curobo_config``).

```bash
python submodules/IsaacLab-Arena/isaaclab_arena/evaluation/policy_runner.py \
  --viz kit \
  --policy_type shape_sorting.curobo_policy.CuroboPolicy \
  --num_steps 200 \
  --external_environment_class_path shape_sorting.shape_sorting_env:ShapeSortingEnvironment \
  shape_sorting_test \
  --embodiment so101_abs_joint
```

Goal XY is placed on the robot-base → ``goal_object`` line, 3 cm toward the
robot from the object, at Z = 10 cm (robot base frame), with tilt/roll = 0
(top-down). Override the object with ``--goal_object <scene_name>``
(default ``shape_piece_cube``).





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

<img width="2000" height="300" alt="shapes_row" src="https://github.com/user-attachments/assets/f97236a6-0cca-4d52-83b5-9f3f5f385347" />


## Building the dataset

### Scripted synthetic dataset generation

This script will use a cuRobo based scripted policy to execute the shape-sorting task and record the demos in LeRobot dataset format.

```bash
python -m shape_sorting.generate_policy_demos \
  --viz kit \
  --task_description "Insert the shapes into the sorting box." \
  --policy_type shape_sorting.curobo_policy.CuroboPolicy \
  --generation_num_trials 50 \
  --max_retries 100 \
  --output_dir ./datasets/curobo_shape_sorting \
  --dataset_repo_id Artefacts/shape-sorting-so101 \
  --push_to_hub \
  --num_success_steps 12 \
  --action_noise 0.01 \
  shape_sorting_test \
  --embodiment so101_abs_joint \
  --debug_key_reset
```

Run `python -m shape_sorting.generate_policy_demos --help` for more options.



https://github.com/user-attachments/assets/d1a73c54-f0c9-4ed2-bb4b-c7765d8c1150



### Teleoperation data collection


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

> TODO: Add instructions for recording demos in LeRobot dataset format.


https://github.com/user-attachments/assets/6e2105bf-f46c-4b04-8061-7cf49cbc7e35

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


## Training and evaluation with LeRobot

You can learn more about LeRobot in the [general docs](https://huggingface.co/docs/lerobot/en/index) and [CLI docs](https://huggingface-lerobot.mintlify.app/).


Train an ACT policy on the shape-sorting dataset:
```bash
lerobot-train \
  --dataset.repo_id=Artefacts/shape-sorting-so101 \
  --policy.type=act \
  --output_dir=outputs/train/act_shape-sorting-so101 \
  --job_name=act-shape-sorting-so101 \
  --policy.device=cuda \
  --wandb.enable=true \
  --job.target=a10g-small \
  --policy.repo_id=Artefacts/act-shape-sorting-so101
```

For more info on training with LeRobot, see the [LeRobot documentation](https://huggingface.co/docs/lerobot/main/en/il_robots#train-a-policy).



Evaluate with the LeRobot CLI
```bash
PYTHONPATH=. lerobot-eval \
  --policy.path=Artefacts/act-shape-sorting-so101 \
  --policy.device=cuda \
  --output_dir ./outputs/eval/act_shape-sorting-so101 \
  --env.discover_packages_path=envhub \
  --env.type=shape_sorting_arena \
  --env.visualizer=kit \
  --rename_map='{"observation.images.camera_ego_rgb": "observation.images.ego_view", "observation.images.external_camera_rgb": "observation.images.exterior_image"}' \
  --eval.batch_size=1 \
  --eval.n_episodes=1
```

### Run the evaluation with Artefacts

Follow these steps to set up your Artefacts project. For more details, refer to the [documentation](https://docs.artefacts.com/getting-started/).

```bash
artefacts run eval
```
The evaluation videos and metrics will show up on your Artefacts dashboard.
