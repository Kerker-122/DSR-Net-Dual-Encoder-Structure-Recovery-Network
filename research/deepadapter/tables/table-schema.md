# Tables

## v2 layout aligned with SPMNet

| Number | Role | Required contents | Current status |
|---|---|---|---|
| 1 | Datasets | Dataset names, train/validation/test, source, annotation | Actual current setup pending |
| 2–3 | Main comparison | Two datasets per table, mIoU/S-alpha/weighted-F/mean-E/MAE if confirmed; method groups and source labels | No values transferred from SPMNet |
| 4 | DeepAdapter and PSD | Fixed-budget adaptation x decoder configurations, precise loss control | New study results pending |
| 5 | PSD components | no SAG/no ICCS, SAG only, ICCS only, full; fixed encoder and supervision | Current results pending |
| 6 | HSAL | Actual Conventional vs HSAL recipes, final metrics, optional BCE-only control | Current results pending |

Figures: qualitative prediction comparison; feature response before/after PSD; ICCS iterations; normalized training loss plus final validation metrics. Iteration set 1/5/10/15 and normalization procedure are reference templates, not claims of completed new experiments. No synthetic measurements generated.

## v1 planning record
Table 1 datasets: name, train/val/test, annotation, low-light subset definition.
Table 2 main results: method, dataset, metric, source, run protocol; no values available.
Table 3 adaptation x PSD: components, metric, parameters, training budget.
Table 4 band/loss/scale ablations; Table 5 efficiency.
No synthetic numbers. Actual aggregation and repeated-run statistics pending.
