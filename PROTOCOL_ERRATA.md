# PROTOCOL ERRATA (FINAL_V1)

The frozen files (`FINAL_V1_PROTOCOL.md`, `configs/FINAL_V1/protocol.json`, the modelling code) are left byte-identical so that their recorded
hashes stay valid. The corrections below were found by the independent audit (`audit_opus/OPUS_AUDIT.md`, 20 October 2026). None of them
changes a split, a feature, a model, a prediction or a metric.

| ID | Where | Frozen text | Correction | Effect on results |
|---|---|---|---|---|
| E-1 | `FINAL_V1_PROTOCOL.md` section 2 (D-003); `src/final_v1/stage_a_build.py` docstring of `canonical_person`; column `age_in_name` in the manifests | "folder name plus sex plus age plus letter-anagram of the name code"; "age typo" | The two-digit number in the file names is the **year of birth (19xx)**, not the age: group means and SDs equal those of the spreadsheet ages after subtracting from the recording year (controls 2017 - 1949.9 = 67.1, SD 5.16 in both; patients 66.8, SD 9.13 in both). The implemented identity rule used **folder name + sex + anagram of the letter code**; the number was not used. The "age typo" of PD_002 is a birth-year typo (52 versus 63 in one file). | None (rule and its three merges unchanged). |
| E-2 | `FINAL_V1_PROTOCOL.md` section 2 | Starred controls not described | The control spreadsheet (`Tab 3.xlsx`) marks four controls with an asterisk and lists three notes: "registrazioni di scarsa qualità" (poor-quality recordings), "no brano/no frasi (difficoltà nella lettura)" (no passage / no phrases, reading difficulty), "difficoltà nella lettura" (reading difficulty). They are HC_001, HC_007, HC_008 and HC_020; all four are included as specified. | None; reported in the error analysis. |
