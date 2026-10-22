# WavLM baseline subset report: versions

Every version is kept in its own folder `versions/v<version>/` (PDF, LaTeX source, build log, Overleaf zip, generator copy).
The generator `src/final_v1/wavlm_subset_report.py` never overwrites an existing version. Unreleased drafts are kept in `versions/_unreleased/`.

| Version | Date | Pages | Changes | PDF SHA-256 (16) | TeX SHA-256 (16) | Generator SHA-256 (16) |
|---|---|---|---|---|---|---|
| 1.0 | 21 Oct 2026 | 9 | Internal draft built from the separate PatchTST proposal run (v4.0). Never delivered. **Superseded by 1.1.** | `f0e47e2cc11ae3bb` | - | - |
| 1.1 | 21 Oct 2026 | 15 | Rebuilt on the FINAL_V1 comparison. LR and XGB from the official frozen run; Random Forest added as exploratory amendment A2 (S11). Full Final Report structure (abstract, motivation, related work, background, methodology, baseline and extensions, results, error analysis, sensitivity, discussion, future, conclusion). v4.0 run only as a labelled cross-check. | `621fd54882481d1f` | `18e458f6b45aade3` | `91048413305de02b` |
| 1.2 | 22 Oct 2026 | 24 | Rewritten to the guideline's Final Report column. New experiments under amendment A3: wide hyperparameter study for LR/RF/XGB (S12, 256 settings) and nested partial fine-tuning of WavLM-large (S13); exact parameter counts (S14). New: training vs validation vs test, failure analysis by task, sensitivity to task family, pre-declared improvement rule, detailed future directions. v4.0 cross-check removed. | `8a4147309a29e59f` | `9d9b65d6d1456c33` | `541885bd878b2a8c` |

Latest version: 1.2 (`versions/v1.2/WAVLM_BASELINE_SUBSET_REPORT_v1.2.pdf`; copy at the workspace root).
