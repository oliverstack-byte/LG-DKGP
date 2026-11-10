# Data acquisition and preprocessing

## PushT and Cube

Obtain the original HDF5 datasets from the [LeWM collection](https://huggingface.co/collections/quentinll/lewm).
Keep the exact file names and layout:

```
data/pusht_expert_train.h5
data/ogbench/cube_single_expert.h5
```

The recorded inventories in `assets/provenance/single_view_data_inventory.json` give
expected file sizes and array shapes. PushT has 2,336,736 action rows and 18,685
episodes; Cube has 2,010,000 action rows and 10,000 episodes, including terminal
nonfinite action rows. The release does not claim a full-file content SHA for
these original large files. Episode selections and lengths are fixed separately.

Use `--set data_root=/absolute/storage` if data lives elsewhere. Training and
validation prior episodes are listed in `assets/manifests/<task>/train.json` and
`validation.json`; evaluations use `evaluation.json`. The prior split is by
episode. The original single-view LeWM backbone retains its original training
window split and sample-standard-deviation action normalization. The prior and
policy use the recorded population-standard-deviation action coordinates.

## Caddy, Scene2, and RoboTwin

Raw source datasets are published at
[CLTRAY/SAGE-data](https://huggingface.co/datasets/CLTRAY/SAGE-data).
Only the datasets and simulator assets are used; no external action model is
required. Training uses `training/<task>_train.tar.part*` with the corresponding
`.parts.json`, plus `training/<task>_validation.tar`.

```bash
uv run --locked --no-sync python tools/download_native_data.py --env libero_caddy
uv run --locked --no-sync python tools/download_native_data.py --env libero_scene2
uv run --locked --no-sync python tools/download_native_data.py --env robotwin_a2b
```

Files belong under `data/sage/source/training`. The reader accesses concatenated
archive parts and their HDF5 members through bounded seekable views; merging or
unpacking is unnecessary. The bundled original part registries provide full
part hashes. The download tool checks them. Supply `--revision` to pin a Hub
revision; identity is established by the recorded bytes/SHA, not a moving branch
name. The published training/validation sources require hundreds of GB; no raw
training dataset is copied into this release.

Native evaluation uses the released `fixtures/` and does not download raw
training archives. The 667 native main queries include camera observations,
reset seeds, scene identities, causal prefixes, and goal geometry. RoboTwin also
includes recorded arm paths for source-state reconstruction. Only a path string
was relocated inside RoboTwin fixture files; every image/state/action array is
unchanged. Original and released fixture hashes are recorded in
`assets/provenance/fixture_relocation.json`.

## Preprocessing and supervision

Images are decoded as RGB, converted to float in [0,1], normalized using ImageNet
mean/std, and resized to 224×224. Native LeWM uses the agent/wrist camera slots
specified by its checkpoint config. RoboTwin live images pass through the
collector-compatible JPEG/color conversion in `prior_data/native_image_codec.py`.
That conversion does not apply to already decoded source images or LIBERO.

Prior caches store one raw-step latent/action sequence per episode. Invalid
terminal action suffixes are dropped; invalid interior actions are rejected.
Native data uses executed commands with stored success and command hashes.
RoboTwin raw visual indices must match action indices; a trailing visual-index
sentinel is not treated as an invented frame. Frameskip five is used only when
packing LeWM action blocks, not by dropping prior history actions.

Training latent normalization uses prior-training episodes only. History windows
contain commands before the prediction start, never future labels. Training
history lengths are sampled causally up to the available context; validation
uses the original deterministic length schedule. The single-view prior uses
the `t+25` observation as its window goal. Native prior training retains its
original terminal-observation conditioning. Future expert commands supervise
the training loss and are never supplied to the online planner.

The planner scale is computed from adjacent standardized commands in successful
training episodes:

```
sigma² = mean_over_action_coordinates(mean_over_training_pairs((a[t+1]-a[t])²)) / 2
```

All command coordinates, including grippers, contribute to this statistic.
Calibration does not use held-out evaluation success. The main evaluation
query manifests retain the previously verified eligibility selections and
per-seed order; do not resample them to compare against the archived results.
