# DATA_AUDIT (Stage A, 2026-10-20T09:37:09Z)

Source: `/root/Projects/PatchTST/Italian Parkinson's Voice and speech` (read-only). Copy: `data_input/`. All 838 copied files have identical SHA-256 to the originals (see `WORKSPACE_MANIFEST.csv`).

## Inventory
- Files found: 831 WAV (+7 spreadsheets). Parsed with one file-name rule (task code, name code, age, sex, date, time): 0 non-matching.
- Included (PD + elderly controls, all 16 tasks): **785 recordings, 47 speakers (25 PD, 22 controls)**, 4.978 audio hours. PD recordings 436, control recordings 349.
- Excluded: 45 young-control files (D-001) and 1 byte-identical duplicate (D-002).
- Sample rates among included files: {'16000': 625, '44100': 160}. All files are mono 16-bit PCM and decode as finite, non-empty audio.
- Class prevalence: 53.2 % PD by speaker, 55.5 % by recording.

## Speaker identity (D-003)
- Raw folders: 28 PD person folders (some sessions repeat people), 22 elderly folders. Name codes within a person folder can differ between sessions by letter order.
- Rule (as implemented): same folder name + sex + same letters in the name code (anagram). *Correction E-1 (audit): an earlier version of this line also listed age; the two-digit number in the file names is the year of birth and was not used by the rule.* Merged people (source keys): 11-16/Nicola S/NSICCHOI/M;6-10/Nicola S/SNCIHCIO/M; 1-5/Roberto R/RRIOVBIE/M;11-16/Roberto R/RROIBVEI/M; 11-16/Vito S/VSIPTIOZ/M;6-10/Vito S/VSIPTIOZ/M
- One speaker (PD_002) has a different two-digit birth-year number in one file (52 versus 63); they are kept as one person. *(Correction E-1: earlier called an 'age typo'.)*
- Result: 47 speakers; no speaker appears under two IDs by this rule; no speaker ID is shared by two groups.
- Speakers with fewer than 16 recordings (missing tasks are not imputed): PD_003 (15), PD_009 (15), PD_010 (10), PD_016 (14), PD_018 (14), HC_020 (13)

## Tasks
Recording counts by task code: {'B1': 49, 'B2': 44, 'D1': 50, 'D2': 50, 'FB1': 47, 'PR1': 50, 'VA1': 50, 'VA2': 49, 'VE1': 50, 'VE2': 49, 'VI1': 50, 'VI2': 49, 'VO1': 50, 'VO2': 49, 'VU1': 50, 'VU2': 49}.
Read-text and dialogue distinction of the prompt: this corpus has no dialogue task; all 16 task codes are pooled as instructed by the project owner. The task code is not a model input.

## Integrity and leakage risks found
1. **Every recording date holds one class only** (dates by label: controls 2017-03-17, 2017-03-21, 2017-03-23, 2017-03-30, 2017-04-02; PD 2016-05-26, 2016-06-16, 2017-01-24, 2017-02-10). Recording session and class are fully confounded.
2. **All 44.1 kHz audio is PD** (160 recordings); native sample rate is a label leak unless everything is resampled (all audio is resampled to 16 kHz).
3. One byte-identical duplicate pair inside one speaker (D-002); cannot cross folds.
4. Speaker folders occur in different recording sessions; merging is required to keep a person in one fold (done).
5. Duplicate checks: byte duplicates 2 (before exclusion), decoded-PCM duplicates after exclusion 0, duplicate sample IDs 0.

## Isolation check
Workspace distinct from original: True. Original audio files changed: 0 of 831. Original project file list changed since the start snapshot: no.

## Additions from the independent audit (20 October 2026)
- **Birth year, not age (E-1).** The two-digit number in the file names is the year of birth. Spreadsheet ages (TAB 5 for patients, Tab 3 for controls): patients 66.8 +/- 9.1 years (40 to 80), 16 men / 9 women (25 unique people; the sheet lists the three people recorded twice twice); controls 67.1 +/- 5.2 years (60 to 77), 10 men / 12 women. Sex in the file names matches the sheets for every group count.
- **Flagged controls (E-2).** Four controls carry an asterisk in Tab 3 with notes on poor-quality recordings and reading difficulty (one has no passage and no phrase recordings): HC_001, HC_007, HC_008, HC_020. All are included.
- **Independent re-derivation.** Re-parsing the raw tree with separate code gives the same 785 files, labels and 47-speaker partition. Two controls share the same letters in the name code (VITO A and VITO L) but differ in birth year, folder and date; they are different people.
- **Near-duplicates across speakers.** Highest cosine similarity of WavLM embeddings between recordings of different speakers is 0.9948 (within speaker 0.9976); no cross-speaker pair reaches 0.999. No cross-speaker duplicate audio is suspected.
