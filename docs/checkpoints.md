# Released checkpoints

The separate release asset bundle contains byte-for-byte copies of the final full prior, its mean-stage checkpoint, the frozen world model, and its model config. The GP and world model are tied by the parent SHA recorded inside the prior. No training was performed to create this bundle.

| Environment | Prior SHA256 (prefix) | Frozen world-model SHA256 (prefix) |
|---|---|---|
| cube | `45ca9dac88918c77` | `6aa6d78dbc39a0ed` |
| libero_caddy | `649d7cab51c68e4d` | `2e2cf2c852ebf095` |
| libero_scene2 | `6bbf1c8f157be6f0` | `71aec1276fc6b657` |
| pusht | `ff0ae7326e959b26` | `8b3a855845a6712a` |
| robotwin_a2b | `35ab96eaaa5e88e6` | `b9cb1d4b771e87c0` |

Full SHA256 and byte sizes are in `assets/checkpoints/manifest.json`. Each `prior.pt` retains its original `model_config`, `model`, `inducing`, metadata, optimizer state, and completion flags. `mean.pt` retains its original pretraining state. Released checkpoints are not renamed internally or converted to another tensor layout.

PushT/Cube use the original independent GP layout; native tasks use the batched layout. `lgdkgp.training.prior.construct_checkpoint` accepts both. Latent normalization buffers are embedded in each prior. The scalar CEM exploration scale is a separate training-derived recipe parameter; it is not a second set of learned weights.

The `prior.pt` files include serialized Python/PyTorch metadata from this project. The release loader deliberately supports this format. To continue an interrupted new training run, preserve both the best output and its `.last` file; final inference checkpoints alone are not advertised as cursor-exact continuation states.

For source hosting, keep weights in a separate model/release artifact, as documented in the README. `.gitignore` prevents accidental ordinary-Git inclusion. The GitHub source contains their manifest/configs and excludes `.pt` binaries.

Checkpoint selection for every training stage, explicit mean initialization, interrupted-run resume, and evaluation is documented in [README.md](../README.md#train-and-choose-checkpoints).
