# Study report versions (FINAL_V1)

Every version is kept unchanged in its own folder (`versions/v<version>/`: PDF, LaTeX source, figures, Overleaf zip and the generator script that produced it).
The generator `src/final_v1/study_report.py` always writes into the folder of its own version and refuses to rebuild version 1.0.

| Version | Date | Pages | Changes | PDF SHA-256 (16) | TeX SHA-256 (16) | Generator SHA-256 (16) |
|---|---|---|---|---|---|---|
| 1.0 | 20 Oct 2026 | 47 | First study edition (FINAL_V1 only; guideline order). Five appendix pages were landscape. | `793b438128b657f2` | `71fe2d9840cd9b81` | `3f8596d4fbddd747` |
| 1.1 | 20 Oct 2026 | 54 | Extension of 1.0: feature dictionary (section 5.1.3), feature-importance tables (section 7.2.5), revision history, all pages A4 portrait (wide appendix tables split into two portrait tables with the same numbers), typo '89%%' fixed. | `6c51e5adc427323a` | `c9ab696a4b4b03f0` | `afe98059b92e498e` |
| 1.2 | 20 Oct 2026 | 57 | Extension of 1.1: importance figure redrawn without overlapping labels (same numbers); new section 7.2.6 'SHAP explanations' (out-of-fold TreeSHAP for XGBoost, linear SHAP for logistic regression; figures and table; analysis S10). | `43ddc74b6eb37117` | `71ba84266f31942c` | `7c74991f821d8229` |
| 1.3 | 20 Oct 2026 | 58 | Extension of 1.2 (earlier text unchanged): new figure 'How an embedding is built' in Representation 2. | `f910b2299f7aa23f` | `817aac6e8b78ea38` | `1e78f172138edc7c` |

Latest version: 1.3 (`versions/v1.3/PROJECT_REPORT_FINAL_V1_v1.3.pdf`; copy at the workspace root).
