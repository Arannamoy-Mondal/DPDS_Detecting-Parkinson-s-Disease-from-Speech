# REPRODUCIBILITY
- Original project: `/root/Projects/PatchTST`; isolated workspace: `/root/Projects/Project4a_FINAL_V1_ISOLATED`.
- Environment: `ENVIRONMENT.md`; Python venv in `.venv`.
- Order of execution (all commands run from the workspace with `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python src/final_v1/<script>`): `stage_a_inventory.py`, `stage_a_build.py`, `splits.py`, `download_models.py`, `acoustic.py`, `extract_embeddings.py`, then the Stage C to F scripts listed in `ARTIFACT_MANIFEST.csv` once they exist.
- Seeds: outer split 42; inner 142+fold; selection 242+fold; XGBoost grid subset 42; bootstrap 42.

## Complete command sequence (all run from the workspace with `CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 .venv/bin/python`)
1. `src/final_v1/stage_a_inventory.py`, `src/final_v1/stage_a_build.py`, `src/final_v1/splits.py`, `src/final_v1/download_models.py`, `src/final_v1/acoustic.py`
2. `src/final_v1/stage_a_docs.py`, `src/final_v1/isolation_check.py stageA`, `src/final_v1/write_mgmt.py stageA`
3. `src/final_v1/freeze_protocol.py` (freeze), `tests/test_pipeline.py` (Stage C), `src/final_v1/write_mgmt.py stageC`
4. `src/final_v1/extract_embeddings.py KEY part 3 3` for KEY in WLM, W2V, XLSR and part 0..2, then `src/final_v1/extract_embeddings.py KEY collect`
5. `src/final_v1/runner.py run <workers> <prefix>` (FV1-AC, FV1-W2V, FV1-XLSR, FV1-WLM, FV1-H)
6. `src/final_v1/aggregate.py`, `src/final_v1/audit_oof.py`, `src/final_v1/group_pdfs.py`, `src/final_v1/write_mgmt.py stageD`, `src/final_v1/isolation_check.py final`, `src/final_v1/write_mgmt.py stageE`, `src/final_v1/write_mgmt.py status`
7. `src/final_v1/report_analysis.py`, `src/final_v1/train_scores.py`, `src/final_v1/final_report.py pdf`, `src/final_v1/build_reports.py`, `src/final_v1/artifact_manifest.py`
8. Audit: `audit_opus/scripts/a01_reextract.py` ... `a05_demographics_and_flags.py`; supplementary analyses: `supplementary/code/*.py`; whitepaper: `supplementary/code/build_whitepaper.py`.
Reproducibility of model fitting is exact under single-threaded BLAS (see ENVIRONMENT.md).
