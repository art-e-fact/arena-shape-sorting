# IsaacLab-Arena shape-sorting environment and SO-101 embodiment

Example workspace for solving a shape-sorting task with an SO-101 arm in a procedurally generated environment.

Main features:
 - Procedurally generated environment
 - SO-101 arm embodiment
 - Scripted synthetic dataset generation for imitation learning
 - Teleoperation and data collection
 - Training and evaluation with LeRobot and Artefacts

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
 - [Development](DEVELOPMENT.md)

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
| `--control_hz` | `30.0` | Env-step rate (Hz), and the fps of datasets recorded from this env. Physics stays near 200 Hz. Keep the CuroboPolicy pacing flags (`--waypoint_stride`, `--close_steps`, `--open_steps`, `--home_steps`) in proportion, or the demos stretch in wall-clock time instead of getting shorter |

`--enable_cameras` is a shared Arena flag (pass it before `shape_sorting_test`), not an env-subcommand option.

<img width="2000" height="300" alt="shapes_row" src="https://github.com/user-attachments/assets/f97236a6-0cca-4d52-83b5-9f3f5f385347" />


## Building the dataset

You can skip this step and use our [pre-generated dataset](https://huggingface.co/datasets/Artefacts/shape-sorting-so101) on Hugging Face.

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
  --grasp_perturb_prob 0.3 \
  --miss_notice_delay_max_steps 45 \
  shape_sorting_test \
  --embodiment so101_abs_joint \
  --debug_key_reset
```

Run `python -m shape_sorting.generate_policy_demos --help` for more options.

**Recovery clips.** One rollout is not one episode. The scripted policy marks a
*checkpoint* after each verified insertion and asks for a *cut* when it catches its
own mistake; a cut throws away the frames since the last checkpoint, saves the
confirmed ones as an episode, and starts a new episode at the failure state. A
recovery episode therefore opens on "gripper closed on nothing" and its actions are
the fix, so the dataset teaches the recovery without teaching the mistake.

The two flags above are what create those failures on purpose, since cuRobo plans
from ground truth and almost never misses on its own:

| Flag | Default | Description |
|------|---------|-------------|
| `--grasp_perturb_prob` | `0` | Probability that the **first** grasp attempt on a piece is offset. Retries use the true pose, so the recorded recovery is correct. |
| `--grasp_perturb_xy_m` | `0.02` | Half-width of the uniform XY offset [m]. Too large and the plan fails outright instead of near-missing — check how many perturbed attempts still plan in the logs. |
| `--miss_notice_delay_max_steps` | `0` | Max steps to carry an empty gripper before noticing, sampled per miss. Non-zero also covers "empty gripper halfway to the bin", which is where a trained policy actually fails. 45 ≈ 1.5 s at 30 fps. |
| `--insert_settle_steps` | `15` | Extra open-gripper steps to wait for the piece to land before declaring the insertion failed (which ends the rollout). |

`--generation_num_trials` still counts **successful rollouts**, so a run exports at
least that many episodes and usually more. Why this matters for training:
[`recovery-gap.md`](training_research.local/recovery-gap.md).

**Adding episodes to an existing dataset.** `--generation_num_trials` counts
successes *for this run*, not the size of the finished dataset, so `--resume`
with the same number appends that many more:

```bash
python -m shape_sorting.generate_policy_demos \
  ... --resume --generation_num_trials 150 ...   # 150 existing -> 300
```

`--resume` and `--overwrite` are mutually exclusive, and resume appends to the
**local** `--output_dir` — it does not pull the dataset back from the Hub, so if
the local copy is gone, download it first. `--push_to_hub` then uploads the whole
dataset, not just the new episodes. Full semantics and the validation rules:
[`dataset.md`](training_research.local/dataset.md).

> **Do not append recovery clips to a dataset you also train a held-out split on.**
> `--dataset.eval_split` holds out the *last* episodes, not a random sample, so
> resuming with grasp perturbation turned on makes the held-out set almost entirely
> recovery clips and the held-out loss stops measuring generalisation. Regenerate the
> dataset with perturbation enabled from the start instead.



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

*See the [SO-101 embodiment](https://github.com/art-e-fact/isaaclab-so101#joint-space-gamepad-layout-so101_abs_joint--gamepad) for more detail.*

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

### Train on a Nebius GPU

`nebius/train.sh` submits training as a one-off Nebius job: it provisions a GPU,
installs LeRobot, runs `lerobot-train`, and releases the GPU when the run ends —
no VM to create by hand and nothing to remember to shut down. `--timeout` is a
hard ceiling, so a hung run cannot quietly bill overnight.

Prerequisites: the [Nebius CLI](https://docs.nebius.com/cli) authenticated
(`nebius auth login`), and somewhere to get `HF_TOKEN` and `WANDB_API_KEY` from.

```bash
cp .env.example .env   # then fill in the two tokens; .env is gitignored
```

Each token is resolved on its own, in this order: the env file (`.env` in the
repo root, or `--env-file PATH`), then the surrounding shell, then a MysteryBox
secret (`creds` by default, `--secret NAME` to change it). A partial
`.env` is fine — whatever is missing falls back. The launcher prints which
source it used for each token.

A token from `.env` or the shell is sent as a plain job environment variable,
which means it is stored in the job spec and readable by anyone with access to
the project; MysteryBox keeps it out of the spec. Fine for a personal token,
worth avoiding for a shared one.

```bash
nebius/train.sh --name smolvla-500ep --timeout 4h -- \
  --policy.path=lerobot/smolvla_base \
  --policy.device=cuda \
  --policy.input_features=null --policy.output_features=null \
  --policy.repo_id=Artefacts/smolvla-shape-sorting-30fps \
  --policy.push_to_hub=true \
  --save_checkpoint_to_hub=true \
  --dataset.repo_id=Artefacts/shape-sorting-so101-30fps \
  --dataset.eval_split=0.1 \
  --batch_size=64 --steps=20000 \
  --save_freq=2000 --eval_steps=500 --log_freq=100 \
  --num_workers=8 --seed=42 --env_eval_freq=0 \
  --wandb.enable=true --wandb.disable_artifact=true \
  --job_name=smolvla-shape-sorting-500ep
```

Everything after `--` is passed to `lerobot-train` untouched. The job disk is
discarded when the job ends, so `--policy.push_to_hub` and
`--save_checkpoint_to_hub` are what make a run recoverable.

`--policy.repo_id` and `--policy.input_features=null
--policy.output_features=null` are not optional: `smolvla_base` inherits
`push_to_hub=true` and the camera keys of its own training set, and a run without
those flags dies at startup. Why, and what the error messages look like:
[`smolvla-tuning.md`](training_research.local/smolvla-tuning.md).

Defaults: `gpu-h100-sxm` / `1gpu-16vcpu-200gb`, 12 h timeout, 250 GiB disk.
Add `--platform gpu-h200-sxm` for 141 GB of VRAM, `--preemptible` for a cheaper
but interruptible GPU, `--follow` to stream logs, or `--dry-run` to validate the
request for free. `nebius/train.sh --help` lists them all.

```bash
nebius ai job list                 # what is running
nebius ai job logs <id> --follow   # stream a run
nebius ai job cancel <id>          # stop paying for it now
nebius ai job ssh <id>             # shell into a running job to debug
```

#### Choosing batch size and steps

**`--batch_size=64`.** Throughput flattens above it (128 buys 8% for double the
VRAM) and SmolVLA's preset learning rate of 1e-4 is tuned for it. Training is
GPU-bound, so `--num_workers=8` is already enough.

**Steps: one epoch is `train_frames / batch_size` steps**, and the run starts
overfitting at roughly 12 epochs. Recompute this whenever the dataset changes —
a step count from a smaller dataset does not transfer.

| Dataset | steps/epoch @ bs64 | ~12 epochs | run to |
| --- | ---: | ---: | ---: |
| 150 episodes | 435 | 5,000 | — |
| 500 episodes (current) | 1,451 | 17,000 | **20,000** |

The 12-epoch elbow was measured at 150 episodes and has not been re-measured at
500, so the command above runs ~15% past it. A curve that visibly turns over
tells you where the elbow is; one that stops at the guess tells you nothing. If
the held-out loss is still falling at the end, `--resume` extends the run.

Checkpoint every 2,000 steps and pick by rollout success rate rather than taking
the last one — the repo root always holds the final step, which is rarely the
best. Loading an intermediate checkpoint back:

```bash
hf download Artefacts/smolvla-shape-sorting-30fps \
  --include "checkpoints/006000/pretrained_model/*" --local-dir ckpt
# then --policy.path=ckpt/checkpoints/006000/pretrained_model
```

Where these numbers come from — the batch-size sweep, the measured loss curve,
and the epoch arithmetic: [`training_research.local/`](training_research.local/index.md).

Evaluate with the LeRobot CLI
```bash
lerobot-eval \
  --policy.path=Artefacts/act-shape-sorting-so101_1 \
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
