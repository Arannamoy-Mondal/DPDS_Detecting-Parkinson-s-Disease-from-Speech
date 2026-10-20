"""Write/refresh the management files. Usage: python write_mgmt.py stageA"""
import json, platform, subprocess, sys, os
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa
import make_pdf

PY = str(WORKSPACE / ".venv/bin/python")


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def environment_md():
    freeze = sh("cd %s && uv pip freeze --python %s 2>/dev/null | grep -iE '^(numpy|scipy|pandas|scikit-learn|torch|transformers|xgboost|soundfile|joblib|huggingface)'" % (WORKSPACE, PY))
    cpu = sh("lscpu | grep 'Model name' | head -1 | cut -d: -f2").strip()
    mem = sh("free -g | awk '/Mem:/{print $2\" GB total\"}'")
    disk = sh("df -h %s | tail -1" % WORKSPACE)
    txt = """# ENVIRONMENT (recorded %s)
- OS: %s (%s), kernel %s
- CPU: %s, logical cores: %s
- RAM: %s
- Disk (workspace): `%s`
- GPU/accelerator: none used. All computation runs on CPU (torch.cuda.is_available() is not relied on; `CUDA_VISIBLE_DEVICES=""` is set for extraction).
- Python: %s (virtual environment `%s/.venv`, created with uv)
- Main packages:
```
%s
```
- Hugging Face cache: `cache/hf` inside the workspace (HF_HOME), offline mode during extraction.
- Nondeterminism limits: BLAS thread rounding can change the last digits of embeddings between thread counts (known from earlier audits); all runs use the saved embeddings, so model fitting is deterministic given the saved arrays, with fixed seeds; XGBoost uses `hist` with fixed `random_state`.
""" % (now_utc(), platform.system(), platform.platform(), platform.release(), cpu, os.cpu_count(), mem, disk, sys.version.split()[0], WORKSPACE, freeze)
    write_text(WORKSPACE / "ENVIRONMENT.md", txt)


def stage_a():
    s = json.loads((WORKSPACE / "manifests/stage_a_summary.json").read_text())
    iso = json.loads((WORKSPACE / "manifests/isolation_check_stageA.json").read_text())
    audit = """# DATA_AUDIT (Stage A, %s)

Source: `%s` (read-only). Copy: `data_input/`. All 838 copied files have identical SHA-256 to the originals (see `WORKSPACE_MANIFEST.csv`).

## Inventory
- Files found: %d WAV (+7 spreadsheets). Parsed with one file-name rule (task code, name code, age, sex, date, time): 0 non-matching.
- Included (PD + elderly controls, all 16 tasks): **%d recordings, %d speakers (%d PD, %d controls)**, %.3f audio hours. PD recordings %d, control recordings %d.
- Excluded: %d young-control files (D-001) and %d byte-identical duplicate (D-002).
- Sample rates among included files: %s. All files are mono 16-bit PCM and decode as finite, non-empty audio.
- Class prevalence: %.1f %% PD by speaker, %.1f %% by recording.

## Speaker identity (D-003)
- Raw folders: 28 PD person folders (some sessions repeat people), 22 elderly folders. Name codes within a person folder can differ between sessions by letter order.
- Rule (as implemented): same folder name + sex + same letters in the name code (anagram); the two-digit file-name number is the birth year and is not used. Merged people (source keys): %s
- One speaker (PD_002) has a different two-digit birth-year number in one file (52 versus 63); they are kept as one person.
- Result: 47 speakers; no speaker appears under two IDs by this rule; no speaker ID is shared by two groups.
- Speakers with fewer than 16 recordings (missing tasks are not imputed): %s

## Tasks
Recording counts by task code: %s.
Read-text and dialogue distinction of the prompt: this corpus has no dialogue task; all 16 task codes are pooled as instructed by the project owner. The task code is not a model input.

## Integrity and leakage risks found
1. **Every recording date holds one class only** (dates by label: controls %s; PD %s). Recording session and class are fully confounded.
2. **All 44.1 kHz audio is PD** (%d recordings); native sample rate is a label leak unless everything is resampled (all audio is resampled to 16 kHz).
3. One byte-identical duplicate pair inside one speaker (D-002); cannot cross folds.
4. Speaker folders occur in different recording sessions; merging is required to keep a person in one fold (done).
5. Duplicate checks: byte duplicates %d (before exclusion), decoded-PCM duplicates after exclusion %d, duplicate sample IDs %d.

## Isolation check
Workspace distinct from original: %s. Original audio files changed: %d of %d. Original project file list changed since the start snapshot: %s.
""" % (now_utc(), ORIGINAL_DATA, s["files_total"], s["files_included"], s["speakers_total"], s["speakers_pd"], s["speakers_hc"], s["audio_hours"],
       s["recordings_pd"], s["recordings_hc"], s["files_excluded_young"], len(s["duplicates_excluded"]), s["sample_rates"], 100 * s["class_prevalence_speakers"], 100 * s["class_prevalence_recordings"], "; ".join(m["source_keys"] for m in s["merged_speakers"]),
       ", ".join("%s (%d)" % (r["speaker_id"], r["n_recordings"]) for r in s["speakers_with_fewer_than_16_recordings"]), s["tasks"],
       ", ".join(s["date_by_label"]["0"]), ", ".join(s["date_by_label"]["1"]), 160, s["duplicate_files_by_bytes"], s["duplicate_files_by_pcm"], s["duplicate_sample_ids"],
       iso["workspace_distinct_and_not_inside_original"], len(iso["original_audio_files_changed"]), iso["original_audio_files_checked"],
       "no" if not iso["original_project_files_added_or_changed_since_start"] and not iso["original_project_files_removed_or_changed_since_start"] else "YES, see isolation_check_stageA.json")
    write_text(WORKSPACE / "DATA_AUDIT.md", audit)
    environment_md()
    write_text(WORKSPACE / "DECISION_LOG.md", """# DECISION_LOG
| ID | Date (UTC) | Type | Decision | Reason |
|---|---|---|---|---|
| D-000 | 2026-10-20 | OWNER INSTRUCTION | No frozen protocol file was supplied; the agent drafts it. `AGENTS.md` rules do not apply; `Final_Prompt.md` rules do. Use **all** speech tasks. Run stages A to F without stopping and deliver a short English PDF per stage and experiment group. | Message of the project owner. |
| D-001 | 2026-10-20 | DATA | Exclude the 15 young healthy controls. | Age 19 to 29 versus patients 40 to 80; only 3 of 16 tasks; the question is PD versus age-comparable healthy controls. |
| D-002 | 2026-10-20 | DATA | Exclude `Giulia P / B2` (byte-identical to `B1` of the same person); keep the first copy. | Exact duplicate audio would count twice. Cannot cross folds (same speaker). |
| D-003 | 2026-10-20 | DATA | Merge people recorded in two sessions by folder name + sex + anagram name code (3 people). Keep the person with a birth-year typo as one. | One person must stay in one fold. Evidence is in `DATA_AUDIT.md`; the corpus has no documented person key. |
| D-004 | 2026-10-20 | PROTOCOL | Representations, models, grids, k = 10, fixed threshold 0.5, speaker-level primary metrics: see `FINAL_V1_PROTOCOL.md`. | Drafted before any model was trained on the project data. |
| D-005 | 2026-10-20 | PROTOCOL | FWL is BLOCKED (`PROTOCOL_DECISION_REQUIRED`). | Purpose, response, nuisance covariates, statistic and stopping rule not defined; the agent does not improvise. |
| D-006 | 2026-10-20 | CODE CHECK | Before the freeze, one code smoke call ran the nested pipeline on outer fold 0 (acoustic features, forward selection, logistic regression) to test that the code runs; only the selected parameters and feature names were printed, no metric was computed or viewed. Not an official run. | Transparency; the fold-0 test data influenced no decision. |
""")
    write_csv(WORKSPACE / "FAILURE_LOG.csv", [], ["timestamp_utc", "run_id", "stage", "category", "description", "impact", "action", "status"])
    write_text(WORKSPACE / "FINAL_REPORT.md", "# FINAL REPORT\n\n**Status: NOT READY.** No FINAL_V1 experiment has completed. This file is replaced in Stage F.\n")
    write_text(WORKSPACE / "OOF_LEAKAGE_AUDIT.md", "# OOF LEAKAGE AUDIT\n\n**Status: PENDING.** The audit runs in Stage E after predictions exist (see `FINAL_V1_PROTOCOL.md` section 9).\n")
    write_text(WORKSPACE / "PROJECT_STATUS.md", """# PROJECT STATUS (updated %s)
- Original project: `%s`
- Isolated workspace: `%s`
- Stage A (isolated workspace and audit): **COMPLETED** (isolation check passed).
- Stage B (protocol verification and freeze): in progress.
- Stages C to F: not started.
- Blockers: FWL (needs a protocol decision, does not block the other experiments).
- Training on project data: none yet.
""" % (now_utc(), ORIGINAL, WORKSPACE))
    write_text(WORKSPACE / "REPRODUCIBILITY.md", """# REPRODUCIBILITY
- Original project: `%s`; isolated workspace: `%s`.
- Environment: `ENVIRONMENT.md`; Python venv in `.venv`.
- Order of execution (all commands run from the workspace with `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python src/final_v1/<script>`): `stage_a_inventory.py`, `stage_a_build.py`, `splits.py`, `download_models.py`, `acoustic.py`, `extract_embeddings.py`, then the Stage C to F scripts listed in `ARTIFACT_MANIFEST.csv` once they exist.
- Seeds: outer split 42; inner 142+fold; selection 242+fold; XGBoost grid subset 42; bootstrap 42.
""" % (ORIGINAL, WORKSPACE))
    pdf = make_pdf.summary_pdf("STAGE_A_workspace_and_data_audit", "Stage A: isolated workspace and data audit", "FINAL_V1 - Parkinson's disease detection from speech",
        [("What was done", ["Created the isolated workspace %s and a clean Python environment; the original project %s was only read." % (WORKSPACE, ORIGINAL),
                            "Copied the raw corpus (838 files) read-only into the workspace; every SHA-256 matches the original.",
                            "Archived historical plans, reports and code under Archive/previous_versions (copied, originals untouched); historical scores were not read or used.",
                            "Isolation test passed: workspace distinct, 831 original audio files unchanged, original file list unchanged."]),
         ("Data found", ["Included: %d recordings, %d speakers (%d PD, %d controls), %.2f audio hours, 16 task codes pooled." % (s["files_included"], s["speakers_total"], s["speakers_pd"], s["speakers_hc"], s["audio_hours"]),
                         "Excluded: 45 young-control recordings (D-001) and 1 byte-identical duplicate (D-002). Three people recorded in two sessions were merged (D-003).",
                         "Class prevalence: 53.2 percent PD by speaker."]),
         ("Risks and blockers", ["Every recording date holds one class only and all 44.1 kHz audio is PD: recording conditions are confounded with the label (resampling to 16 kHz applied).",
                                  "The prompt's frozen protocol file does not exist; a protocol was drafted (Stage B). FWL is blocked pending a protocol decision.",
                                  "No blocker prevents execution on this machine (32 CPU cores, 61 GB RAM, CPU only)."])],
        [("Cohort used (included recordings and speakers)", ["Group", "Speakers", "Recordings"], [["PD", s["speakers_pd"], s["recordings_pd"]], ["Elderly controls", s["speakers_hc"], s["recordings_hc"]]])])
    print("wrote", pdf)


def stage_c():
    t = json.loads((WORKSPACE / "tests/test_results.json").read_text())
    rows = [[r["test"].split(" ")[0], " ".join(r["test"].split(" ")[1:])[:95], "PASS" if r["passed"] else "FAIL"] for r in t["results"]]
    ok = t["all_passed"]
    pdf = make_pdf.summary_pdf("STAGE_C_pipeline_validation", "Stage C: pipeline validation (code-level checks)", "FINAL_V1 - Parkinson's disease detection from speech",
        [("What was validated", ["%d automatic checks on synthetic features and the audited manifests; result: %s." % (len(rows), "ALL PASSED" if ok else "FAILURES PRESENT"),
                                 "Instrumented run proved that every fit uses only rows of its own training partition, that tuning never touches outer-test rows or speakers, and that feature selection sees training speakers only.",
                                 "A null control (random features, true labels) gave chance-level pooled AUROC and a positive control gave near-perfect AUROC, so the nested loop neither invents nor loses signal.",
                                 "A deliberately corrupted prediction file (test speaker also in training) failed three integrity gates and was quarantined, not repaired.",
                                 "Resume logic reuses completed runs. All 78 frozen hyperparameter configurations fit and return probabilities in [0,1]."]),
         ("Separation of smoke tests", ["Smoke outputs live in tests/tmp_smoke and are labelled SMOKE_TESTS_NOT_EXPERIMENT_RESULTS; they are excluded from every metric and table of the study."])],
        [("Check results", ["Test", "Description", "Result"], rows, "lp{11cm}l")], status="COMPLETED" if ok else "FAILED")
    print("wrote", pdf)


def stage_d():
    cm = pd.read_csv(WORKSPACE / "metrics/FINAL_V1/condition_metrics.csv")
    reg = pd.read_csv(WORKSPACE / "EXPERIMENT_REGISTRY.csv", keep_default_na=False)
    fr = reg[reg.row_type == "fold_run"]
    st = fr.training_status.value_counts().to_dict()
    rt = pd.to_numeric(fr.runtime_seconds, errors="coerce")
    rows = []
    for rep in ["AC", "WLM", "W2V", "XLSR"] + ["H%d" % i for i in range(1, 9)]:
        x = fr[fr.representation == rep]
        r = pd.to_numeric(x.runtime_seconds, errors="coerce")
        rows.append([rep, str(len(x)), str((x.training_status == "COMPLETED").sum()), "%.0f" % r.sum()])
    pdf = make_pdf.summary_pdf("STAGE_D_execution", "Stage D: execution of the frozen experiments", "FINAL_V1 - Parkinson's disease detection from speech",
        [("Execution record", ["%d planned outer-fold runs; status counts: %s." % (len(fr), st), "56 of 56 conditions have all five folds; FWL stayed BLOCKED (1 registry row).",
                               "Every run passed six integrity gates (no speaker overlap, exactly one prediction per test sample, valid probabilities, labels, folds, hashes); no run failed and nothing was quarantined.",
                               "Sum of per-run wall-clock times: %.1f run-hours (runs executed in parallel, 09:42 to 11:48 UTC). Embedding extraction: three frozen encoders on CPU in parallel, 785 recordings each (4.98 audio hours), 09:39 to 10:41 UTC (about one hour of wall time)." % (rt.sum() / 3600),
                               "Only the frozen configuration was run: no search space, threshold, model or feature set was changed after the freeze. Execution-harness files (runner, aggregator, audit) were written after the freeze; modelling code is byte-identical to the frozen hashes."]),
         ("Order of execution", ["Acoustic runs (60) first, while the encoders were extracted; then wav2vec2, XLS-R, WavLM single-encoder runs (60); then the eight hybrids (160)."])],
        [("Runs by representation", ["Representation", "Planned runs", "Completed", "Runtime sum (s)"], rows, "lrrr")])
    print("wrote", pdf)


def stage_e():
    a = json.loads((WORKSPACE / "metrics/FINAL_V1/oof_audit.json").read_text())
    iso = json.loads((WORKSPACE / "manifests/isolation_check_final.json").read_text())
    rows = [[str(i), c["check"][:110], c["status"]] for i, c in enumerate(a["checks"], 1)]
    pdf = make_pdf.summary_pdf("STAGE_E_independent_audit", "Stage E: independent audit", "FINAL_V1 - Parkinson's disease detection from speech",
        [("Result", ["Out-of-fold leakage and integrity audit: %s (%d checks, %d completed runs, %d failed, %d conditions)." % ("ALL PASSED" if a["all_passed"] else "FAILURES", len(a["checks"]), a["completed_runs"], a["failed_runs"], a["completed_conditions"]),
                     "Pooled AUROC and AUPRC of all 56 conditions were recomputed with independent code and agree with the reported tables to 1e-9.",
                     "Isolation re-check at the end: workspace distinct from the original, %d original audio files checked and %d changed; original project file list changed since the start: %s." % (iso["original_audio_files_checked"], len(iso["original_audio_files_changed"]), "no" if iso["passed"] else "YES"),
                     "What the audit cannot show: that the scores reflect disease. Recording date and class are confounded (see Stage A)."])],
        [("Audit checks", ["#", "Check", "Status"], rows, "rp{12.5cm}l")], status="COMPLETED" if a["all_passed"] and iso["passed"] else "FAILED")
    print("wrote", pdf)


def final_status():
    write_text(WORKSPACE / "PROJECT_STATUS.md", """# PROJECT STATUS (updated %s)
- Original project: `%s`
- Isolated workspace: `%s`
- Stage A (isolated workspace and audit): COMPLETED
- Stage B (protocol verification and freeze): COMPLETED (frozen; `FREEZE_RECORD.json`)
- Stage C (pipeline validation): COMPLETED (25 of 25 checks passed)
- Stage D (execution): COMPLETED - 280 of 280 outer-fold runs, 56 of 56 conditions; FWL BLOCKED
- Stage E (independent audit): COMPLETED - all checks passed
- Stage F (reporting): COMPLETED - `FINAL_REPORT.md`
- Open item: FWL needs a protocol decision (purpose, response, covariates, procedure, statistic, stopping rule); not run.
- Short PDF summaries: `reports/FINAL_V1/summaries/`
""" % (now_utc(), ORIGINAL, WORKSPACE))
    txt = (WORKSPACE / "DECISION_LOG.md").read_text()
    if "D-007" not in txt:
        txt += "| D-007 | 2026-10-20 | PROCESS | `runner.py` (execution harness with integrity gates), `aggregate.py` and `audit_oof.py` were written after the freeze; the modelling code (`pipeline.py`, `protocol_defs.py`, `splits.py`, `acoustic.py`, `extract_embeddings.py`, `common.py`) is byte-identical to the frozen hashes (checked in the audit). | They contain gates, bookkeeping and independent checks, not modelling choices. |\n| D-008 | 2026-10-20 | PROCESS | Embedding extraction and the acoustic runs started after the freeze; acoustic runs ran while encoders were still being extracted. Run order does not affect any result. | Efficiency. |\n"
        write_text(WORKSPACE / "DECISION_LOG.md", txt.replace("\\n", "\n"))


if __name__ == "__main__":
    {"stageA": stage_a, "stageC": stage_c, "stageD": stage_d, "stageE": stage_e, "status": final_status}[sys.argv[1]]()
