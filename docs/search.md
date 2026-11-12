# Expert-scaled search

The two search modes share the full neural-plus-GP continuation mean and the
task recipe's `sigma`, which is the expert-derived **standard deviation**.
Both load exactly the `prior` and `world_model` paths selected by the recipe
or explicit `--set` overrides. No checkpoint conversion or refitting is needed
to switch search modes.

| Mode | Initial action covariance |
| --- | --- |
| `isotropic` (default) | `sigma**2 * I` |
| `gp_covariance` | `sigma**2 * K_hat / mean(diag(K_hat))` |

Here `K_hat = c * K_function + floor_std**2 * I`, using the GP's residual
**function** covariance, without likelihood noise. Each recipe sets
`planning.gp_covariance_scale=2.0` and
`planning.gp_covariance_floor_std=0.1`, corresponding to `c` and `floor_std`.
These planner settings do not overwrite older checkpoint model configurations.
The covariance uses time-major `(forecast step, action coordinate)` ordering.
Normalization matches the average initial marginal variance to the expert
`sigma**2`; it preserves the relative variances and correlations of `K_hat`.

CEM factors the normalized covariance once per planning call and retains that
factor while refitting diagonal perturbations in whitened coordinates. Numerical
Cholesky jitter is tried in order: 0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3. Plan records
include the selected mode, expert sigma, pre-jitter average variance, jitter,
and effective average variance. The input checkpoint is never modified.

## Evaluate and select checkpoints

After installing the appropriate simulator extra as described in the README:

```bash
# Same weights and expert sigma, isotropic search.
uv run --locked --no-sync python -m lgdkgp eval --config configs/tasks/pusht.json \
  --set planning.covariance_mode=isotropic \
  --set output=outputs/pusht_isotropic

# Same weights and expert sigma, normalized GP-covariance search.
uv run --locked --no-sync python -m lgdkgp eval --config configs/tasks/pusht.json \
  --set planning.covariance_mode=gp_covariance \
  --set output=outputs/pusht_gp_covariance

# Explicitly choose another completed GP-stage prior and its paired world model.
uv run --locked --no-sync python -m lgdkgp eval --config configs/tasks/pusht.json \
  --set planning.covariance_mode=gp_covariance \
  --set prior=/path/to/gp.pt \
  --set world_model=/path/to/paired/world_model.pt \
  --set output=outputs/pusht_custom_gp_covariance
```

The same options work with `cube.json`, `libero_caddy.json`,
`libero_scene2.json`, and `robotwin_a2b.json`. Use the matching simulator
extra and a fresh output path. All five recipes specify `H=P=25`, a 75-command
budget, history capacity 32, and 300 candidates / 30 iterations / 30 elites.
Their expert standard deviations are:

| Environment | Expert sigma |
| --- | ---: |
| PushT | 0.3629534581292668 |
| Cube | 0.3493684570257794 |
| Caddy | 0.5723290840773352 |
| Scene2 | 0.6532117084286896 |
| RoboTwin | 0.026753395532356745 |

## Training and evaluation protocol

New feature preparation and prior training use the goal latent at `t+25` in
all five environments. Future expert commands `t` through `t+24` supply the
training targets. Caches with a terminal-goal policy must be prepared again
under a fresh output directory; existing model checkpoints remain loadable.

PushT and Cube use their specified simulator-state goal predicates. Caddy,
Scene2, and RoboTwin require the manipulated object and evaluated end effectors
to match their recorded goal positions within 0.03 m. RoboTwin's native terminal
flag is recorded as a diagnostic and does not independently award success.
The README describes preparation, both training stages, validation-based
checkpoint selection, and explicit evaluation checkpoint overrides.
