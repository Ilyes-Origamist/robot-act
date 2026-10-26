

# Robot Learning for Object Manipulation with ACT on MuJoCo

## Setup and Installation

For portability to Google Colab and Kaggle (for training), `uv` is used instead of Anaconda. 

Install `uv`:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Install dependencies (ffmpeg for video decoding; GL/EGL libraries for headless MuJoCo):
```bash
sudo apt install -y git ffmpeg libgl1 libegl1 libosmesa6 build-essential cmake
```
If using Windows WSL, make sure to intall `libevdev`:
```bash
sudo apt install libevdev-dev
uv pip install evdev
```

Clone the repository:
```bash
git clone https://github.com/Ilyes-Origamist/robot-act
cd robot-act
```

Do not create or activate a virtual environment yourself: `uv sync` creates
`.venv/` from `uv.lock`. Deactivate conda first if it is active (`conda deactivate`).

### Option A: local machine, CPU only

```bash
uv sync --extra cpu
```

Use this for development, data generation and evaluation.


### Option B: local machine, NVIDIA GPU

Requires an NVIDIA driver supporting CUDA ≥ 12.8 (check "CUDA Version" in `nvidia-smi`).
The CUDA toolkit (`nvcc`) is **not** needed; the PyTorch wheels bundle their own runtime.

For CUDA version 12.8 -- 12.9
```bash
uv sync --extra cu128
```
FOr CUDA ≥ 13.0
```bash
uv sync --extra cu130
```

Verify:

```bash
source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Run project commands after `source .venv/bin/activate` or `uv run --no-sync <command>`.


### Option C: Google Colab (T4 GPU)

Select **Runtime → Change runtime type → T4 GPU**, then run in a cell:

```python
!python --version   # must be 3.12 or newer
!git clone https://github.com/Ilyes-Origamist/robot-act
%cd robot-act
!pip install -q uv
!uv pip install --system -r requirements-colab.txt   # keeps Colab's preinstalled CUDA torch
import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

On Colab, call tools directly (`!lerobot-train ...`) instead of `uv run`; there is no `.venv`.

### Optional: headless rendering

MuJoCo needs a rendering backend to produce camera images. Enable headless (offscreen)
rendering on servers, over SSH, on Colab, or whenever you don't need a window.
Set it **before** MuJoCo is imported.

| Setup | Command |
|---|---|
| Local, NVIDIA GPU (current shell) | `export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl` |
| Local, CPU only (current shell) | `export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa` |
| Google Colab (first cell) | `import os; os.environ["MUJOCO_GL"] = "egl"; os.environ["PYOPENGL_PLATFORM"] = "egl"` |

To open the interactive viewer, use a window backend instead:
`MUJOCO_GL=glfw uv run python -m mujoco.viewer`.

### Verify and Test

If venv is active:
```bash
python scripts/check_installation.py
```
Without activating the venv:
```bash
uv run --no-sync python scripts/check_installation.py
```

Training smoke test:
```bash
lerobot-train \
  --policy.type=act \
  --dataset.repo_id=lerobot/aloha_sim_insertion_human \
  --env.type=aloha --env.task=AlohaInsertion-v0 \
  --batch_size=12 --steps=2000 \
  --env_eval_freq=1000 --eval.n_episodes=2 --eval.batch_size=1 \
  --save_freq=1000 --log_freq=50 --num_workers=4 \
  --policy.device=cuda --policy.use_amp=true \
  --policy.push_to_hub=false \
  --wandb.enable=false \
  --job_name=act_insertion_smoke \
  --output_dir=outputs/train/act_insertion_smoke
```

Check memory usage on another terminal:
```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 1
```