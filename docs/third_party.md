# Attribution

Original LG-DKGP contributions are licensed under the root `LICENSE`, copyright
2026 oliverstack and ShaoBo Yang and DeepTouchLab.

- LeWM architecture and training utilities are derived from the original
  LeWorldModel repository. Its MIT license and Lucas Maes copyright are retained
  in `third_party/licenses/LeWM_LICENSE`. `src/lgdkgp/models/world/jepa.py` and `blocks.py` retain the original architecture needed
  to load released checkpoints.
- `stable-worldmodel` is an external dependency at commit
  `15a5538d492ae524c64cb18cc56a2d70611e877e`; its license remains upstream.
- Native LIBERO code/assets remain upstream. A copy of the MIT license is in
  `third_party/licenses/LIBERO_LICENSE`; runtime source is downloaded at the recorded commit.
- Bundled RoboTwin source retains `third_party/robotwin/LICENSE`. Original source
  identities and the recorded-path adaptation are retained in `third_party/setup`.
  Asset notices are in `third_party/setup/robotwin/ASSETS_NOTICE.md`.
- The native demonstration data are distributed by CLTRAY/SAGE-data. Their
  distribution terms and original simulator/asset attributions remain applicable.
- Large model weights and query fixtures are organized as separate release
  artifacts. `assets/checkpoints/manifest.json` and the query manifests record their identities.
