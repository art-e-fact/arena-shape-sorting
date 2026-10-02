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
    - [cuRobo edge-case videos on Artefacts](#curobo-edge-case-videos-on-artefacts)
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
(``so101_abs_joint``). The cuRobo robot config ships with ``arena-so101``.

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
| `--teleop_device` | none | Teleop device (`keyboard`, `so101_gamepad`, `spacemouse`, `so101_leader`, …) |
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
| `--box_mass` | `0.35` | Sorting box mass (kg). Light enough that pulling a wedged piece out drags it; see "Pieces that start in the wrong place" for measurements |
| `--clearance` | `0.003` | Hole clearance around each piece (m) |
| `--edge_chamfer` | `0.001` | Piece top/bottom edge chamfer (m) |
| `--hole_chamfer` | `0.001` | Hole rim lead-in chamfer (m) |
| `--drop_on_hole_prob` | `0.0` | Probability per episode that one random piece starts dropped onto its own lid hole — tilted, yawed and off-centre — instead of on the table. It falls in, wedges, or lands on the lid: the states a trained policy leaves after letting go in the wrong place. For dataset generation; keep it `0` for evaluation |
| `--tip_over_prob` | `0.0` | Probability per episode that one random piece starts lying on its side where it stood (never the cube, which on its side is the same cube). The policy picks it up and sets it down standing. For dataset generation; keep it `0` for evaluation |
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
  --place_perturb_prob 0.3 \
  shape_sorting_test \
  --embodiment so101_abs_joint \
  --drop_on_hole_prob 0.2 \
  --tip_over_prob 0.2 \
  --debug_key_reset
```

Run `python -m shape_sorting.generate_policy_demos --help` for more options.

**How a piece is placed.** The arm stops at a hover pose above the matched hole, lowers
to the release pose, opens the jaws, and then climbs back out along the same path before
anything is planned again. The climb-out is the part that matters: planning straight from
the release pose can start inside the box's collision margin, and cuRobo then fails every
attempt on the spot — which looks like the arm freezing after a release.

The hole frame sits on the lid top, so with the default box (`0.04` tall) and pieces
(`0.03` tall) the descent closes 25 of the 30 mm between the two poses, and the piece is
released with its bottom still about **5 mm above the lid**:

| | piece origin above the hole frame | piece bottom vs. lid top |
|---|---|---|
| end of the transport | +45 mm | 30 mm clear |
| end of the descent (release) | +20 mm | 5 mm clear |

That remaining gap is the likeliest reason a piece sometimes fails to fall in.
`--place_z_offset_m 0.015` puts the bottom exactly on the lid, below that it is already
inside the hole at release.

| Flag | Default | Description |
|------|---------|-------------|
| `--place_z_offset_m` | `0.02` | Piece-origin height above the hole at release [m]. The piece drops the last few mm and the rim chamfer aligns it. |
| `--place_hover_z_offset_m` | `0.045` | Piece-origin height at the end of the transport [m]. The gap to the line above is the descent, and the retreat replays it backwards — so this is the climb-out height. |
| `--grasp_height_m` | `-0.002` | TCP height (between the jaw tips) above the piece's origin for the grasp [m]. Sets where on the piece the jaws close (default: fingertips 5 mm below its centre). Measured from the piece's live pose, and raised as needed to keep the fingertips off the lid. Replaces the absolute `--grasp_z_m 0.145`. |

Two things were tried here and measured *worse* or neutral, so they are not the defaults —
`training_research.local/insert-strategy.md` has the numbers. Seating the piece into the
lid before releasing (`--place_z_offset_m 0.011`) quadrupled failed insertions, because the
jaws are rigid and jam a slightly misaligned piece where dropping lets gravity correct it.
Gripping higher (`--grasp_height_m 0.004`) made no measurable difference.

> When comparing two configs, pass `--placement_seed N`. `--seed` does **not** control
> object placement, so without it the two runs see different scenes.

**Recovery clips.** One rollout is not one episode. The scripted policy marks a
*checkpoint* after each verified insertion and asks for a *cut* when it catches its
own mistake; a cut throws away the frames since the last checkpoint, saves the
confirmed ones as an episode, and starts a new episode at the failure state. A
recovery episode therefore opens on "gripper closed on nothing" and its actions are
the fix, so the dataset teaches the recovery without teaching the mistake.

**How the policy is structured.** Every motion is one of three: *go* (a planned move to
a hover pose, then a straight descent to contact), *jaw* (open or close in place) and
*up* (replay the descent backwards). Between motions, one decision function looks at the
world — is something in the jaws, is the piece in the box, is it tilted — and picks the
next goal: insert it, park it, grasp it, move on to the next piece, or put this one to
the back of the queue. The arm only ever plans from a hover pose, never from one with
its fingers around something, which is what used to freeze it after a release or a
missed grasp. The rules behind this are written at the top of
[`curobo_policy.py`](arena_envs/src/shape_sorting/curobo_policy.py).

**Grasping.** The wrist rolls about the gripper's own axis, which is vertical over a
piece, so the jaws can close along any direction. The policy lines them up with the
piece's flat faces — never two corners — and tries each such direction, starting from
the straight base-to-piece line the grasp was tuned with. A piece whose every direction
is blocked (say, squeezed between two neighbours) goes to the back of the queue and is
tried again once another piece has been moved; a full round with no progress ends the
rollout instead of freezing it.

A **missed grasp** is noticed when the jaw closes all the way on nothing; the arm backs
off and re-approaches the piece.

A **failed insertion** is noticed either just before the jaws open — the held piece is
measured against its hole for XY offset, yaw (folded into the piece's own symmetry) and
tilt — or afterwards, when it never turns up inside the cavity. In the second case the
arm closes the jaws again first, which needs no planning because the piece is still
between them. Both cases then lift the piece clear and **re-measure the grasp**, which
is what turns the miss into a correction: the next attempt aims from where the piece
actually sits in the jaws rather than from where it was when it was first picked up.

Tilt then picks the recovery, because tilt is the one error re-aiming over the hole does
not fix:

- **upright** — place again, re-aimed. No parking, no re-grasp.
- **tilted** — set it down standing on a free table spot — straight back where it was
  lifted from if that was the table, otherwise next to the box — pick it up level, place
  again.

"Standing" is the wrist's job: the grip fixes the piece's axis in the gripper, so
pitching the wrist by the same angle stands it up, and choosing the piece's yaw keeps
that pose inside the 5-DoF arm's reach. If it does not plan, the piece is set down
leaning 30° (it still falls onto its base), and last with the gripper upright, which
levels anything tipped under ~45° and nothing steeper. A tilted gripper sits ~10 cm
nearer the robot than the piece it holds, so these set-downs use spots at least 24 cm
out; next to the table, 60° of wrist tilt is about the limit, so a piece lying flat
is nearly always set down leaning.

So a piece is grasped where it lies whatever its pose — tipped against the rim of its
hole, the box or a neighbour, stuck in its hole, or lying on its side on the table — and
stood back up. Standing on its top counts as standing — every piece is a prism, the same
solid upside down — and the cube is standing on any face, being as tall as it is wide.

**Pieces that start in the wrong place.** The policy's own inserts almost never leave a
piece stuck in its hole, because it checks the alignment before letting go. A trained
policy does leave pieces like that. `--drop_on_hole_prob` (an env flag) recreates that
state at reset instead: one piece is let go of over its own hole, off-centre and yawed,
and physics decides whether it falls in, catches the rim and wedges, or lands on the
lid. The policy needs nothing special for it: it waits for the piece to land, deals with
whatever starts on the box first, and takes it out and puts it back like any other
piece. Nothing about that is a mistake, so nothing is cut — the whole recovery is
recorded. `--tip_over_prob` does the same for a piece knocked onto its side; those go
first too, since a lying cylinder brushed by the arm rolls away.

Measured with `--box_mass 1.0`, standing pieces up: `--tip_over_prob 1.0` finished 5/6
rollouts, `--drop_on_hole_prob 1.0` 7/8 (it was 1/3 before, when a piece that came out of
its hole tipped past ~45° landed on its side and stayed there). The misses: a lying
cylinder that rolled out of reach, and a regrasp right against the box wall that closed
on the wall too and lifted the box.

Pulling a stuck piece out drags the box with it. On five paired rollouts with a piece
dropped every episode, the default 0.35 kg box drifted 13–21 mm per rollout and 2/5
rollouts finished; at `--box_mass 1.0` it drifted 0–18 mm and 4/5 finished — partly
because a light box also shifts when the dropped piece hits its rim, rolling the round
piece off. Whatever you pick, use the same value for generation and evaluation.

The flags below create these failures on purpose, since cuRobo plans from ground truth
and almost never misses on its own:

| Flag | Default | Description |
|------|---------|-------------|
| `--grasp_perturb_prob` | `0` | Probability that the **first** grasp attempt on a piece is offset. Retries use the true pose, so the recorded recovery is correct. |
| `--grasp_perturb_xy_m` | `0.02` | Half-width of the uniform XY offset [m]. Too large and the plan fails outright instead of near-missing — check how many perturbed attempts still plan in the logs. |
| `--miss_notice_delay_max_steps` | `0` | Max steps to carry an empty gripper before noticing, sampled per miss. Non-zero also covers "empty gripper halfway to the bin", which is where a trained policy actually fails. 45 ≈ 1.5 s at 30 fps. |
| `--insert_settle_steps` | `15` | Extra open-gripper steps to wait for the piece to land before declaring the insertion failed. A piece that is inside the cavity but still bouncing counts as a success. |
| `--place_perturb_prob` | `0` | Probability that a piece's **first** placement is misaligned on purpose. Retries use the true pose. |
| `--place_perturb_xy_m` | `0.012` | Half-width of the uniform XY offset on a perturbed placement [m]. |
| `--place_perturb_yaw_rad` | `0.79` (45°) | Half-width of the uniform yaw offset [rad]. Yaw is the error a trained policy actually makes on non-circular pieces, and a cube arriving 45° off its hole is the recovery it most needs to have seen. |
| `--max_insert_retries` | `2` | Failed inserts per piece before it is parked and sent to the back of the queue. It gets a fresh budget when it comes back round. |
| `--max_grasp_retries` | `5` | Grasp attempts per piece before it goes to the back of the queue. A grasp direction that does not plan costs nothing. |
| `--approach_height_m` | `0.04` | Height of the hover above each grasp pose; every grasp comes straight down from it and leaves the same way. |
| `--insert_align_xy_tol_m` | `0.005` | Piece-to-hole XY offset that still counts as insertable [m]. Measured: clean inserts release at 0.2–1.8 mm, and a 6.7 mm release failed to drop in. Every insert logs its measured alignment, so each run adds to that sample. |
| `--insert_align_yaw_tol_rad` | `0.17` (10°) | Yaw tolerance [rad]. A 30 mm square in a 3 mm-clearance hole physically binds at about 13°. |
| `--insert_align_tilt_tol_rad` | `0.35` (20°) | Tilt the arm cannot work around [rad]. A guess, deliberately generous — a slightly tipped piece still drops in, and no re-aim can level one. Answers three questions with one number: release or not, re-aim or park, and which faces of a tipped piece are still vertical enough to grip. |

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

*See the [SO-101 embodiment](https://github.com/art-e-fact/isaaclab-so101#joint-space-gamepad-layout-so101_abs_joint--so101_gamepad) for more detail.*

SE(3) differential gamepad:
```bash
  ...
  --embodiment so101_ik \
  --teleop_device so101_gamepad
```

Joint-space gamepad (absolute joints — recommended for SO-101):

```bash
  ...
  --embodiment so101_abs_joint \
  --teleop_device so101_gamepad
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
Add `--platform gpu-h200-sxm` for 141 GB of VRAM, `--follow` to stream logs, or
`--dry-run` to validate the request for free. `nebius/train.sh --help` lists them all.

#### Spot pricing

`--preemptible` runs on a spot-priced GPU that the platform can take back at any
time (SIGTERM, 60 s, SIGKILL). Pair it with `--supervise`:

```bash
nebius/train.sh --preemptible --supervise --name smolvla-500ep --timeout 4h -- ...  # same args as above
```

Inside the job, `nebius/train_sigterm.py` turns the SIGTERM into one more checkpoint
push, so a preemption loses at most one step (or one `--save_freq` interval, if 60 s
is not enough for the 1.3 GB upload). `--supervise` keeps the launcher attached: when
the job ends `FAILED` — Nebius reports a preemption, a timeout and a crash the same
way — it resubmits with `--resume`, which downloads the latest `checkpoints/<step>`
from `--policy.repo_id` and continues sample-exactly in the same W&B run. It gives up
after two attempts in a row that added no checkpoint, so a crash does not loop, and
`--timeout` becomes a per-attempt ceiling. Ctrl-C stops the watcher only; to
continue a run whose watcher died, re-run the same command with `--resume`. Both
need `--save_checkpoint_to_hub=true`.

#### Reusing a model repo id (Hub versioning)

Resume takes the highest `checkpoints/<step>` in the repo, whoever wrote it, so a
fresh run refuses to start while another run's checkpoints are still there. Before
training into a repo id again, freeze the current `main` as a tag, then clear the
checkpoints and their numeric step tags (the tag keeps them reachable):

```python
from huggingface_hub import HfApi
api, repo = HfApi(), "Artefacts/smolvla-shape-sorting-30fps"
api.create_tag(repo, tag="run-005-v3", repo_type="model")
api.delete_folder("checkpoints", repo_id=repo, repo_type="model", commit_message="clear run-005 checkpoints")
for ref in api.list_repo_refs(repo, repo_type="model").tags:
    if ref.name.isdigit():
        api.delete_tag(repo, tag=ref.name, repo_type="model")
```

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

### cuRobo edge-case videos on Artefacts

`curobo_scenarios` records one viewport video per edge case of the scripted cuRobo policy:
pieces starting in a hole or on their side, missed grasps, misaligned inserts, and exhausted
retry budgets. The scenarios are defined in
[`scenarios.py`](arena_envs/src/shape_sorting/scenarios.py). It takes several minutes per
scenario, so it is meant for reviewing the videos on the dashboard, not for routine checks.

```bash
source setup.sh
artefacts run curobo_scenarios
# one scenario, without Artefacts:
python -m shape_sorting.scenarios 10_in_hole --out outputs/scenarios/10_in_hole
```

A scenario fails if it crashes, hangs, writes no video, or leaves a piece unsorted. The one
exception to the last rule is `16_budgets_exhausted` (`expect="completes"`), which is built
to leave pieces unsorted.
