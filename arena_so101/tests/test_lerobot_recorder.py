from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from arena_so101.lerobot.recorder import (
    ACTION_KEY,
    CAMERA_FEATURES,
    STATE_KEY,
    SO101LeRobotRecorder,
    so101_dataset_features,
)


class _FakeDataset:
    def __init__(self, *, root: Path, repo_id: str, fps: int, features: dict) -> None:
        self.root = root
        self.repo_id = repo_id
        self.fps = fps
        self.features = features
        self.num_episodes = 0
        self.frames = []
        self.finalized = False

    @classmethod
    def create(cls, repo_id, fps, *, root, features, **_kwargs):
        return cls(root=root, repo_id=repo_id, fps=fps, features=features)

    @classmethod
    def resume(cls, repo_id, *, root, **_kwargs):
        return cls(
            root=root, repo_id=repo_id, fps=50, features=so101_dataset_features()
        )

    def add_frame(self, frame):
        self.frames.append(frame)

    def has_pending_frames(self):
        return bool(self.frames)

    def save_episode(self):
        self.num_episodes += 1
        self.frames.clear()

    def clear_episode_buffer(self):
        self.frames.clear()

    def finalize(self):
        self.finalized = True


class _FakeVideoEncodingManager:
    def __init__(self, dataset) -> None:
        self.dataset = dataset

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.dataset.finalize()


@pytest.fixture
def fake_lerobot(monkeypatch):
    datasets_module = ModuleType("lerobot.datasets")
    datasets_module.LeRobotDataset = _FakeDataset
    datasets_module.VideoEncodingManager = _FakeVideoEncodingManager
    monkeypatch.setitem(sys.modules, "lerobot.datasets", datasets_module)


def _observation(image_shape=(480, 640, 3)):
    cameras = {
        key: np.full((1, *image_shape), 0.5, dtype=np.float32)
        for key in CAMERA_FEATURES
    }
    return {
        "policy": {"joint_pos": np.arange(6, dtype=np.float64)[None]},
        "camera_obs": cameras,
    }


def test_feature_schema_uses_so101_joint_order():
    features = so101_dataset_features()

    assert features[STATE_KEY]["names"] == [
        "Rotation",
        "Pitch",
        "Elbow",
        "Wrist_Pitch",
        "Wrist_Roll",
        "Jaw",
    ]
    assert features[ACTION_KEY]["shape"] == (6,)
    assert set(CAMERA_FEATURES.values()) <= features.keys()


def test_records_and_commits_normalized_transition(tmp_path, fake_lerobot):
    recorder = SO101LeRobotRecorder(
        root=tmp_path / "dataset",
        repo_id="local/test",
        fps=50,
    )

    snapshot = recorder.snapshot_observation(_observation())
    recorder.add_transition(
        snapshot,
        np.arange(6, dtype=np.float64)[None],
        task="Sort the shapes.",
    )

    frame = recorder.dataset.frames[0]
    assert frame[STATE_KEY].dtype == np.float32
    assert frame[ACTION_KEY].shape == (6,)
    for feature_key in CAMERA_FEATURES.values():
        assert frame[feature_key].dtype == np.uint8
        assert frame[feature_key].shape == (480, 640, 3)

    recorder.save_episode()
    recorder.close()

    assert recorder.num_episodes == 1
    assert recorder.dataset.finalized


def test_close_discards_uncommitted_episode(tmp_path, fake_lerobot):
    recorder = SO101LeRobotRecorder(
        root=tmp_path / "dataset",
        repo_id="local/test",
        fps=50,
    )
    recorder.add_transition(
        recorder.snapshot_observation(_observation()),
        np.zeros((1, 6), dtype=np.float32),
        task="Sort the shapes.",
    )

    recorder.close()

    assert not recorder.dataset.frames
    assert recorder.num_episodes == 0
    assert recorder.dataset.finalized


def test_existing_output_requires_explicit_mode(tmp_path, fake_lerobot):
    root = tmp_path / "dataset"
    root.mkdir()

    with pytest.raises(FileExistsError, match="--resume.*--overwrite"):
        SO101LeRobotRecorder(root=root, repo_id="local/test", fps=50)


def test_writes_reopenable_lerobot_dataset(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "hf_cache"))
    from lerobot.datasets import LeRobotDataset

    root = tmp_path / "dataset"
    image_shape = (64, 64, 3)
    with SO101LeRobotRecorder(
        root=root,
        repo_id="local/test",
        fps=10,
        streaming_encoding=False,
        image_shape=image_shape,
    ) as recorder:
        for _ in range(2):
            recorder.add_transition(
                recorder.snapshot_observation(_observation(image_shape)),
                np.zeros((1, 6), dtype=np.float32),
                task="Sort the shapes.",
            )
        recorder.save_episode()

    dataset = LeRobotDataset("local/test", root=root)
    assert dataset.num_episodes == 1
    assert dataset.num_frames == 2
