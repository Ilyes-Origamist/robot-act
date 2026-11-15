#!/usr/bin/env python3
"""
Random rollouts for the AlohaTransferCube-v0 environment.
"""
# refer to: https://github.com/huggingface/gym-aloha#quickstart
import os
# set EGL rendering (before importing gym) since the dataset was collected with EGL classical rendering
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from pathlib import Path
import gymnasium as gym, gym_aloha, imageio, numpy as np

out = Path("outputs/random_rollouts"); out.mkdir(parents=True, exist_ok=True)
env = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")
print("obs space:", env.observation_space)
print("action space:", env.action_space)

for ep in range(3):
    observation, info = env.reset(seed=ep)                 # the seed fixes the cube's starting pose
    frames, total = [env.render()], 0.0
    while True:
        # sample a random action
        action = env.action_space.sample()
        # take a step in the environment  
        observation, reward, terminated, truncated, info = env.step(action)
        # render the environment and store the frame
        frames.append(env.render()); total += reward
        if terminated or truncated:
            break
    imageio.mimsave(out / f"ep{ep}.mp4", np.stack(frames), fps=50)
    print(f"episode {ep}: {len(frames)-1} steps, return {total}, info {info}")
env.close()