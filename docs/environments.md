# Simulator setup

Run from the checkout root on Linux x86-64. Each `uv sync` selects one simulator
stack; all training/evaluation uses a single visible GPU. Native cases run
serially in fresh processes. Core prior training from prepared shards needs no
simulator extra (`uv sync --locked`).

## PushT and Cube

```bash
uv sync --locked --extra pc
export CUDA_VISIBLE_DEVICES=0
uv run --locked --no-sync python -m lgdkgp eval --config configs/tasks/pusht.json
```

The pinned stable-worldmodel commit registers `swm/PushT-v1` and `swm/OGBCube-v0`.
Use the original HDF5 files in [docs/data.md](data.md). The adapter keeps seeded dataset
reset, native success, the action buffer, and the 50-command evaluation budget.

## LIBERO: Caddy and Scene2

```bash
uv sync --locked --extra libero
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/libero
git -C third_party/libero checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
uv pip install --python .venv/bin/python --no-deps -e third_party/libero
export CUDA_VISIBLE_DEVICES=0
uv run --locked --no-sync python -m lgdkgp eval --config configs/tasks/libero_caddy.json
```

`--no-sync` preserves the separate editable LIBERO installation. Its required
Python dependencies are selected by the `libero` extra, including MuJoCo 2.3.7,
robosuite 1.4.0, and bddl 1.0.1. Keep upstream scene assets, BDDL files, and init
states. The adapter writes absolute task-local paths under the new output.
Control uses Panda OSC pose at 20 Hz and 128×128 agent/wrist cameras before
224×224 model preprocessing. Select another compatible source checkout with
`--set runtime.libero_root=/absolute/LIBERO`.

## RoboTwin A2B

```bash
uv sync --locked --extra robotwin
```

Matching task/runtime source is in `third_party/robotwin`. The extra pins NumPy
1.26.4, SAPIEN 3.0.0b1, and MPlib 0.2.1. Download the ordered asset parts from
[CLTRAY/SAGE-data](https://huggingface.co/datasets/CLTRAY/SAGE-data) at revision
`c95b57fc1293adc8d9364163c2db49ac264e8bd3`, using the registry in
`third_party/setup/robotwin/asset_archive.json`. Concatenate the asset parts in their
registry order into `robotwin_a2b_assets.tar.gz`, then install:

```bash
uv run --locked --no-sync python tools/install_robotwin_assets.py \
  --archive /storage/robotwin_a2b_assets.tar.gz
export CUDA_VISIBLE_DEVICES=0
uv run --locked --no-sync python -m lgdkgp eval --config configs/tasks/robotwin_a2b.json
```

The installer verifies the archive SHA, installs portable bootstrap/kinematics
configs, and refuses to overwrite an existing assets directory. The archive is
about 11.6 GB and unpacked assets about 12.3 GB. Asset notices are retained in
`third_party/setup/robotwin/ASSETS_NOTICE.md`.

Recorded-state reconstruction uses bundled replay-only arm paths and the
original MPlib TOPP/gripper interpolation. Evaluation does not generate new
expert paths. It sends 14D joint/gripper commands and records actual executed
commands for subsequent history. Use `--set runtime.robotwin_root=/absolute/path`
for another compatible runtime; `LEWM_ROBOTWIN_RENDER_DEVICE` selects the SAPIEN
render-device alias. Rendering and CUDA must use the same physical GPU.

Recorded upstream environment YAMLs in `third_party/setup/` document the original
simulator stacks; install this release using the locked extras above. Native
rendering also needs the system graphics/Vulkan/EGL libraries and simulator assets.
