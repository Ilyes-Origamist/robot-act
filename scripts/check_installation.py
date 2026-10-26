#!/usr/bin/env python3
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")          # CPU offscreen rendering
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import torch, mujoco

print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available())   # expect +cpu, False
x = torch.randn(256, 256); print("torch matmul OK", (x @ x).shape)

xml = "<mujoco><worldbody><light pos='0 0 3'/><geom type='box' size='.1 .1 .1'/></worldbody></mujoco>"
m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
for _ in range(100): mujoco.mj_step(m, d)
print("mujoco", mujoco.__version__, "physics OK, t =", round(d.time, 3))
r = mujoco.Renderer(m, 240, 320); r.update_scene(d)
print("render OK", r.render().shape)
r.close()

import gymnasium as gym, gym_aloha
env = gym.make("gym_aloha/AlohaInsertion-v0"); env.reset(seed=0)
print("ALOHA env OK")
env.close()