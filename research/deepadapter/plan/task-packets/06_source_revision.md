# Source-grounded revision v2

User supplied three local PDFs and explicitly assigned formula sources: SPMNet decoder/loss and experiment organization, NightAdapter DeepAdapter, UNeXT other architecture. This authorizes formula-level revision and coordinated updates to all five existing sections. No new model implementation or experiments requested.

Inputs: sources/manifest.json, extracted pages and visually inspected formula pages; existing chapters and user confirmed two-branch four-scale interface.

Deliverables: revised chapters, manuscript_zh_v2.md, source-to-equation map, remaining consistency notes, experiment table schema. Preserve v1.

Method decisions: implement paper-level mathematical description, not claim implementation verification. Reference NightAdapter Eq.4–6 with dimension-consistent notation; distinguish borrowed form from unconfirmed parameter choices. Use UNeXT dense glue construction as source-based version of four-scale auxiliary features and explicitly flag if user's modified code differs. Use SPMNet Eq.8 (weight outside ReLU) as requested paper-based revision, recording conflict with supplied old code. HSAL targets refined prediction gradient, not directly learned shape head; no invented C or S_gt.

Experiment organization: Implementation Details; Datasets; Evaluation Metrics; Comparative Experiments (quantitative/qualitative); Ablations (DeepAdapter addition, PSD, iteration visualization, HSAL). No SPMNet numbers reused as current results. Its training settings and dataset identity are not automatically current settings.

Checks: visually read formula pages, verify equation dimensions, 5 chapters present, numbered equations unique, references resolve, archive v1; evidence and implementation gaps remain explicit.
