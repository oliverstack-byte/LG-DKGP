# Rethinking Actions: Continuity-Aware Priors for World-Model Planning

Single-GPU training and evaluation for **LG-DKGP**: an action-history encoder,
neural mean, and GP residual used with a frozen LeWM and CEM planner.
The reproduction target is our full method on PushT, Cube, Caddy, Scene2, and
RoboTwin. All commands below run from this repository root.

## Install with uv

Linux x86-64, Python 3.10, and one CUDA GPU are supported. The Python version,
dependencies, and upstream world-model commit are recorded in `.python-version`,
`pyproject.toml`, and `uv.lock`.

```bash
uv sync --locked --extra pc       # PushT and Cube
# For a different simulator, select exactly one extra instead:
# uv sync --locked --extra libero
# uv sync --locked --extra robotwin
export CUDA_VISIBLE_DEVICES=0
```

The three simulator extras are mutually exclusive because they require different
MuJoCo/NumPy stacks. Select an extra with `uv sync` before working on that task.
`uv run --locked --no-sync` below uses the environment just installed, including
any separately installed native simulator source. Do not use `--all-extras`.
See [docs/environments.md](docs/environments.md) for LIBERO source and RoboTwin assets.
The pinned PyTorch wheel needs a compatible NVIDIA driver; no multi-GPU or DDP
launcher is supported. The visible device is always addressed as `cuda:0`.

## Data and released weights

The GitHub source contains code, recipes, fixed episode/query manifests, and
licenses. Large checkpoints and native scene fixtures are separate release
assets. Raw training datasets are downloaded separately; see [docs/data.md](docs/data.md).

Place the asset directories at these paths, preserving filenames:

```text
checkpoints/<task>/prior.pt       # final full prior
checkpoints/<task>/mean.pt        # mean pretraining checkpoint
checkpoints/<task>/world_model.pt # paired frozen LeWM
checkpoints/<task>/config.json    # LeWM architecture config
fixtures/<native-task>/...        # fixed native evaluation inputs
```

For the local asset bundle prepared alongside this checkout:

```bash
cp -a ../lg-dkgp-release-assets/checkpoints/. checkpoints/
cp -a ../lg-dkgp-release-assets/fixtures .
uv run --locked --no-sync python tools/verify_artifacts.py
```

The `.pt` files and fixtures are ignored by Git. Download URLs should point to
the separately published release/model assets; this source does not assume an
unpublished URL. [docs/checkpoints.md](docs/checkpoints.md) records checkpoint identities.

## Evaluate

Each task recipe selects its released full prior and paired LeWM explicitly:

```bash
uv run --locked --no-sync python -m lgdkgp eval --config configs/tasks/pusht.json
uv run --locked --no-sync python -m lgdkgp eval --config configs/tasks/cube.json
uv run --locked --no-sync python -m lgdkgp eval --config configs/tasks/libero_caddy.json
uv run --locked --no-sync python -m lgdkgp eval --config configs/tasks/libero_scene2.json
uv run --locked --no-sync python -m lgdkgp eval --config configs/tasks/robotwin_a2b.json
```

Run only tasks supported by the currently selected simulator extra. To choose
other compatible checkpoints, override **both paths explicitly** as needed:

```bash
uv run --locked --no-sync python -m lgdkgp eval --config configs/tasks/pusht.json \
  --set prior=/path/to/gp.pt \
  --set world_model=/path/to/paired/world_model.pt \
  --set output=outputs/pusht_custom_eval
```

Evaluation loads exactly `prior` and `world_model`; it does not search for the
newest file. `prior` must be a completed GP-stage checkpoint, **not** `mean.pt`
or an interrupted `.last` state. Keep the architecture config beside the LeWM
weights. A prior is tied to its training backbone and normalization; do not mix
checkpoints from different tasks or world models. Existing evaluation outputs
are not overwritten: choose a fresh `output`.

Results are written to `<output>/evaluation/`, including `binding.json` with
checkpoint/query hashes, per-query outcomes, and `summary.json` with seed means
and sample SDs. The released five-seed recipes use 101, 202, 303, 404, and 505.

## Train and choose checkpoints

Obtain raw data first. A task recipe's `world_model` selects the frozen LeWM used
for feature extraction. `output` selects the new training directory:

```bash
# PushT / Cube: uv sync --locked --extra pc
uv run --locked --no-sync python -m lgdkgp prepare --config configs/tasks/pusht.json
uv run --locked --no-sync python -m lgdkgp train --config configs/tasks/pusht.json --stage mean
uv run --locked --no-sync python -m lgdkgp train --config configs/tasks/pusht.json --stage gp

uv run --locked --no-sync python -m lgdkgp prepare --config configs/tasks/cube.json
uv run --locked --no-sync python -m lgdkgp train --config configs/tasks/cube.json --stage mean
uv run --locked --no-sync python -m lgdkgp train --config configs/tasks/cube.json --stage gp


# Caddy / Scene2: uv sync --locked --extra libero (plus LIBERO setup)
uv run --locked --no-sync python -m lgdkgp prepare --config configs/tasks/libero_caddy.json
uv run --locked --no-sync python -m lgdkgp train --config configs/tasks/libero_caddy.json --stage mean
uv run --locked --no-sync python -m lgdkgp train --config configs/tasks/libero_caddy.json --stage gp

uv run --locked --no-sync python -m lgdkgp prepare --config configs/tasks/libero_scene2.json
uv run --locked --no-sync python -m lgdkgp train --config configs/tasks/libero_scene2.json --stage mean
uv run --locked --no-sync python -m lgdkgp train --config configs/tasks/libero_scene2.json --stage gp


# RoboTwin: uv sync --locked --extra robotwin (plus asset setup)
uv run --locked --no-sync python -m lgdkgp prepare --config configs/tasks/robotwin_a2b.json
uv run --locked --no-sync python -m lgdkgp train --config configs/tasks/robotwin_a2b.json --stage mean
uv run --locked --no-sync python -m lgdkgp train --config configs/tasks/robotwin_a2b.json --stage gp

```

| Operation | Checkpoint selection |
|---|---|
| `prepare` | Loads the task's `world_model` path; it stays frozen. |
| `train --stage mean` | Starts the mean from initialization; saves the lowest-validation-MSE model to `<output>/priors/mean.pt`. |
| `train --stage gp` | Initializes from `<output>/priors/mean.pt`; saves the lowest-validation-joint-NLL model to `<output>/priors/gp.pt`. Neural parameters are also updated. |
| `train --stage both` | Runs mean, then GP, using the newly selected `mean.pt`. This is the default stage. |
| `train --stage gp --mean-checkpoint PATH` | Uses that completed, matching mean checkpoint instead of `<output>/priors/mean.pt`. Architecture, data identity, normalization, and backbone must match. |
| `train --resume` | Reads `<output>/priors/<stage>.pt.last`, retaining the historical best `<stage>.pt`. Use the original stage, settings, data, and output directory. |
| `eval` | Loads `prior` and `world_model` from the task JSON or explicit `--set` overrides. |

The published final checkpoint is `checkpoints/<task>/prior.pt`; newly trained
full checkpoints are named `outputs/<task>/priors/gp.pt`. They have compatible
state-dict formats. Training never overwrites released weights.

Examples of selecting the mean initializer and resuming a GP fit:

```bash
uv run --locked --no-sync python -m lgdkgp train --config configs/tasks/pusht.json \
  --stage gp --mean-checkpoint /path/to/completed/mean.pt

uv run --locked --no-sync python -m lgdkgp train --config configs/tasks/pusht.json \
  --stage gp --resume
```

If an explicit mean initializer was used, pass the same `--mean-checkpoint`
when resuming. Keep **both** the best `.pt` and `.pt.last` files; the final
released weights alone do not provide an exact interrupted-training cursor.
`gp.pt` becomes evaluation-ready after the stage's stopping policy finishes.

To evaluate newly trained weights, select `gp.pt`. Also set sigma to the value
in `<output>/calibration.json` if the data or action normalization changed:

```bash
uv run --locked --no-sync python -m lgdkgp eval --config configs/tasks/pusht.json \
  --set prior=outputs/pusht/priors/gp.pt \
  --set output=outputs/pusht_trained_eval
# Add --set sigma=<value from outputs/pusht/calibration.json> when needed.
```

The default sigma is the statistic for the released dataset/coordinates.
The combined commands below read the freshly calculated sigma explicitly.

## Combine the commands

The same explicit commands apply to all five environments. Choose `task` from
`pusht`, `cube`, `libero_caddy`, `libero_scene2`, or `robotwin_a2b`, install its
simulator extra, and obtain the raw data and evaluation assets first. The block
below trains a backbone, extracts features with **that exact backbone**, trains
the prior, then evaluates the new prior with its paired backbone and calibration.
Use a fresh run directory; it is passed to every training step.

```bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
task=pusht  # cube, libero_caddy, libero_scene2, or robotwin_a2b
recipe="configs/tasks/${task}.json"
run_dir="outputs/${task}_from_scratch"

# 1. Train the LeWM; select its final configured epoch.
uv run --locked --no-sync python -m lgdkgp train-world --config "$recipe" \
  --set "output=$run_dir"
world_ckpt=$(uv run --locked --no-sync python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["weights"])' \
  "$run_dir/world/complete.json")

# 2. Extract features using the selected LeWM, not the released default.
uv run --locked --no-sync python -m lgdkgp prepare --config "$recipe" \
  --set "output=$run_dir" --set "world_model=$world_ckpt"

# 3. Fit the mean and then the GP; GP reads this run's best mean.pt.
uv run --locked --no-sync python -m lgdkgp train --config "$recipe" \
  --set "output=$run_dir" --stage mean
uv run --locked --no-sync python -m lgdkgp train --config "$recipe" \
  --set "output=$run_dir" --stage gp

# 4. Read training-only calibration and select the new full prior explicitly.
sigma=$(uv run --locked --no-sync python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["sigma"])' \
  "$run_dir/calibration.json")
uv run --locked --no-sync python -m lgdkgp eval --config "$recipe" \
  --set "prior=$run_dir/priors/gp.pt" --set "world_model=$world_ckpt" \
  --set "sigma=$sigma" --set "output=$run_dir"
```

To keep the released frozen LeWM, omit step 1 and use
`world_ckpt="checkpoints/${task}/world_model.pt"` before steps 2–4. The
`train-world` command selects the final epoch in the backbone recipe; it does
not perform validation-best checkpoint selection. Its selected path and hash
are recorded in `<run_dir>/world/complete.json`. Evaluation records the exact
selected checkpoint paths/hashes in `<run_dir>/evaluation/binding.json`.

## Configuration

Task recipes are `configs/tasks/<task>.json`; prior architecture/optimizer settings
are `configs/priors/<task>.json`; backbone settings are
`configs/world/<task>.json`. Use `--set` for task fields and `--prior-set` for
prior training fields:

```bash
uv run --locked --no-sync python -m lgdkgp eval --config configs/tasks/robotwin_a2b.json \
  --set planning.horizon=10 --set planning.execution=10 \
  --set planning.history_limit=1 --set 'seeds=[101]' \
  --set output=outputs/robotwin_h10_l1

uv run --locked --no-sync python -m lgdkgp train --config configs/tasks/pusht.json \
  --stage both --prior-set training.batch_size=64
```

The main recipes use horizon/execution 25/25, a 50-command budget, and CEM with
300 candidates, 30 iterations, and 30 elites. History is executed action history;
`L=0` masks it entirely while retaining visual conditioning. The released priors
support horizons up to 25 in five-command blocks. Architecture changes require
compatible new weights. [docs/data.md](docs/data.md) explains normalization, preprocessing,
query eligibility, and train-only sigma calibration.

## Project structure

```text
src/lgdkgp/
  cli.py                 Command parsing; python -m lgdkgp uses the same entry
  models/prior/          History encoder, neural mean, and GP residual
  models/world/          LeWM backbone and multi-view adaptation
  data/                  Feature preparation and episode/window readers
  training/              Prior fitting, backbone fitting, and training callbacks
  planning/              CEM candidate generation and selection
  evaluation/            Evaluation runner, task loops, and checkpoint checks
  envs/                  Native simulator adapters and success predicates
  common/                Configuration, paths, hashes, and run identity
configs/
  tasks/                 Five task entry recipes
  priors/                Prior architecture and optimization
  world/                 Five backbone training recipes
  evaluation/            PushT/Cube simulator evaluation settings
  environments/          Native environment settings
  normalization/         Recorded action normalization
assets/                  Fixed split/query manifests and checkpoint metadata
third_party/             Simulator source, setup specifications, and licenses
docs/                    Data, environment, checkpoint, and attribution guides
tests/                   CPU and package-compatibility checks
tools/                   Data download, asset installation, and result summaries
```

Large weights and fixtures are distributed separately and restored to
`checkpoints/` and `fixtures/`. `assets/checkpoints/` contains their tracked
metadata, not weights. All local recipe/resource paths resolve from the checkout
root; `LGDKGP_ROOT` can select a checkout when using a non-editable installation.
The compatibility layer resolves original LeWM module names inside released
checkpoint configs; model tensors and state-dict keys are unchanged.

## Checks

```bash
uv run --locked --no-sync python -m pytest -q -p no:cacheprovider
uv run --locked --no-sync python -m lgdkgp check --config configs/tasks/pusht.json --device cpu
```

`check` requires the released asset files and validates SHA/state-dict loading
and a CPU forward pass; it does not run training or a simulator episode.
The code and third-party components retain the licenses in
[LICENSE](LICENSE) and [docs/third_party.md](docs/third_party.md).

## Acknowledgements

This codebase is based on the LeWorldModel (LeWM) repository and was developed
with assistance from OpenAI Codex. Upstream licenses and attribution are retained
in [third_party/licenses/LeWM_LICENSE](third_party/licenses/LeWM_LICENSE) and
[docs/third_party.md](docs/third_party.md).
