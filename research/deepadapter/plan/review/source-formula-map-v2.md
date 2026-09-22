# Formula source map and consistency decisions

All PDFs were read locally; original files remain untouched. Source hashes and page counts are in sources/manifest.json. Page numbers below are PDF page indices starting at 1.

| Manuscript content | Source location | Decision |
|---|---|---|
| SAM2 adapter | UNeXT p2 §2.1; local SAM2UNeXT.py | MLP–GELU–MLP–GELU residual before Hiera block; bottleneck 32 is source config, not automatically current model config |
| Four-scale DINO alignment and glue | UNeXT p3 §2.3–2.4; SAM2UNeXT.py lines around 143–153 | One auxiliary feature projected/resized to four scales; 1x1 concatenation reduction; Hiera-L (144,288,576,1152), DINO-L 1024 and fusion128 belong to reference instantiation |
| DST | NightAdapter p3 Eq1 | Express invertible matrix transform with explicit dimensions; do not copy apparent indexing ambiguities blindly or call channel-token DST plain image FFT |
| Low-frequency token interaction | NightAdapter p4–5 Eq4–6 | Same similarity, token projection and residual fusion; use selected-band width d_L and token shape M×d_L to make products defined; this notation normalization is not a new module |
| Remaining frequency bands | NightAdapter p5 Eq10 versus user's selected submodule | Full source adapts all bands; user's rest-band handling not confirmed. Display a clearly stated identity-bypass working definition, flag for code verification, not assert it is observed implementation |
| SAG | SPMNet p9 Eq11–12 | Conv1x1 → Conv7x7 → sigmoid → pointwise modulation; do not assert depthwise conv because formula does not establish it |
| ICCS | SPMNet p8–9 Eq5–10 and Fig3 | W multiplies OUTSIDE ReLU per supplied paper. Prior code put W in gradient term inside ReLU. v2 follows paper; implementation must be reconciled |
| HSAL | SPMNet p10 Eq13–17 | BCE(refined mask) + lambda mean C(-normalized gradient · S_gt); no invented extra IoU or direct S_pred–S_gt penalty |
| S_gt and C | SPMNet p10 text | Generation method and functional form absent; retain only two necessary formula placeholders rather than claim complete reproducibility |
| Experiments | SPMNet p11–18 §4.1–4.5.3 | Borrow organization, 5-metric template and component/iteration/loss analyses; no values transferred |

## Additional audit findings

1. NightAdapter declares C×N full features but Eq4 selects bands then multiplies full-C tokens; selected widths/orientations are not fully exposed in the printed notation. Manuscript uses d_L and transposed patch-major matrix consistently. Frequency ordering in that paper must not be assumed identical to an arbitrary DST library. Original [768,1024) low group applies only to its stated ordering/C=1024 setting.
2. User diagram's C labels cannot all mean concatenation inside the adapter: paper Eq4–6 require matrix multiplication and addition. Figure labels should later distinguish these operations; no figure edited this turn.
3. SPMNet HSAL shape term is evaluated on normalized gradients of refined P, not a direct loss on predicted S. If ICCS remains inside no_grad without separate supervision, HSAL on refined P cannot train the desired pathway. The PDF alone does not establish that code was fixed.
4. The PDF invokes star-shape prior but does not by itself show that arbitrary learned S and these updates enforce star convexity for all predictions. Manuscript describes shape-guided refinement, without a hard guarantee or convergence proof.
5. SPMNet's iteration T=1,5,10,15 and selected T=10 are reference experimental settings/results. New paper must use its own iteration study. Its normalized training-loss curves do not, by themselves, prove that different objectives yield superior final segmentation or faster convergence to equivalent accuracy.
6. Two branches can each PROVIDE four scales without DINOv2 having four native-resolution stages. User's earlier drawn four-stage pyramid is interpreted as fusion-interface outputs in the UNeXT-based writing version, explicitly pending code alignment.

7. The dimension-consistent low-frequency equations use sqrt(d_L) for the actual interaction width. Original NightAdapter prints sqrt(C) and full-C tokens alongside selected bands. Exact equivalence is not asserted: if its actual implementation projects selected bands back to C, then the interaction width must be set to C and that projection documented. Temperature and token dimensions must be matched to the actual code before finalizing; this rewrite is not a verified replacement implementation.
8. After reading Eq4–9, the core similarity/token-projection/residual steps in the sensitive and insensitive branches are structurally analogous. The sensitive branch additionally performs randomization. The paper should not claim the low-frequency branch alone uniquely invents token interaction or a wholly different residual block.

No source-PDF instructions were treated as user instructions. The papers are evidence and formula references, not authorization to change training code or disclose the anonymous manuscript externally.
