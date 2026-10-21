# WavLM baseline subset report: versions

Every version is kept in its own folder `versions/v<version>/` (PDF, LaTeX source, build log, Overleaf zip, generator copy).
The generator `src/final_v1/wavlm_subset_report.py` never overwrites an existing version. Unreleased drafts are kept in `versions/_unreleased/`.

| Version | Date | Pages | Changes | PDF SHA-256 (16) | TeX SHA-256 (16) | Generator SHA-256 (16) |
|---|---|---|---|---|---|---|
| 1.0 | 21 Oct 2026 | 9 | Internal draft built from the separate PatchTST proposal run (v4.0). Never delivered. **Superseded by 1.1.** | `f0e47e2cc11ae3bb` | - | - |
| 1.1 | 21 Oct 2026 | 15 | Rebuilt on the FINAL_V1 comparison. LR and XGB from the official frozen run; Random Forest added as exploratory amendment A2 (S11). Full Final Report structure (abstract, motivation, related work, background, methodology, baseline and extensions, results, error analysis, sensitivity, discussion, future, conclusion). v4.0 run only as a labelled cross-check. | `621fd54882481d1f` | `18e458f6b45aade3` | `91048413305de02b` |

Latest version: 1.1 (`versions/v1.1/WAVLM_BASELINE_SUBSET_REPORT_v1.1.pdf`; copy at the workspace root).
