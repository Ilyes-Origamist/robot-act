#!/usr/bin/env python3
"""
Inspect a LeRobot scripted dataset for the cube transfer task.
"""
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("lerobot/aloha_sim_transfer_cube_scripted")
print(f"episodes={ds.num_episodes} frames={ds.num_frames} fps={ds.fps}")
for name, feat in ds.features.items():
    print(f"{name:30s} {feat['dtype']:8s} {feat['shape']}")

lengths = np.bincount(np.array(ds.hf_dataset["episode_index"]))
print("episode lengths: min", lengths.min(), "max", lengths.max())

sample = ds[0]                                  # decodes one video frame
for k, v in sample.items():
    print(k, getattr(v, "shape", v))