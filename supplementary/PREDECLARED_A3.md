# Pre-declared settings: amendment A3 (hyperparameter study and WavLM fine-tuning)

Declared on 22 October 2026, before any of these models was fitted, at the owner's request (WavLM baseline subset report,
version 1.2). Exploratory supplementary analyses under the same rules as A1 and A2: written under `supplementary/`,
labelled exploratory, and no frozen FINAL_V1 number changes.

Common to both studies
- People and folds: the 47 FINAL_V1 people, outer folds from `SPLIT_MANIFEST.csv` (seed 42), inner folds from the frozen
  `assign_folds` with seed 142 + outer fold (three speaker-disjoint inner folds inside each outer-training partition).
- Person score = mean of the person's recording probabilities; decision threshold 0.5 (fixed).
- Selection score = speaker-level AUPRC pooled over the three inner validation folds (the FINAL_V1 rule); ties go to the
  earliest configuration in the order listed here.
- Test folds are never used for any choice.

## S12: hyperparameter study (LR, RF, XGB on the frozen WavLM vectors)

Grids (full factorial, listed order = tie order):
- Logistic regression (frozen FINAL_V1 implementation, standardisation on training rows, liblinear):
  C in {1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000} x penalty in {l1, l2} x class_weight in {None, balanced} = 32.
- Random Forest (`RandomForestClassifier`, random_state 42):
  n_estimators in {100, 300, 1000} x max_depth in {2, 4, 8, None} x min_samples_leaf in {1, 5} x max_features in {sqrt, 0.1}
  x class_weight in {None, balanced} = 96.
- XGBoost (frozen FINAL_V1 implementation, hist, random_state 42; min_child_weight 1, reg_lambda 1, scale_pos_weight 1):
  max_depth in {1, 2, 4, 6} x learning_rate in {0.01, 0.03, 0.1, 0.3} x n_estimators in {100, 300} x subsample in {0.7, 1.0}
  x colsample_bytree in {0.3, 1.0} = 128.

Outputs
1. Nested result with the wide grid (honest): each outer fold selects its own configuration by the inner rule.
2. Landscape (descriptive only, never used to choose): every configuration refitted on each outer-training partition and
   scored on the outer test fold; pooled speaker metrics per configuration; one-parameter-at-a-time slices and heatmaps.
3. Training-partition scores of the selected configurations (refit on outer-training rows, scored on the same rows).

## S13: partial fine-tuning of WavLM-large

Input: the same audio recipe as FINAL_V1 (`load_audio_16k`, 10-second windows with the last window aligned to the file end,
the checkpoint's own feature extractor normalisation). Pinned checkpoint `microsoft/wavlm-large`
revision c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c.

Model
- Layers 0 to 23-k frozen; the top k transformer layers, the final encoder layer norm and a new head are trained.
- To save compute, the output of frozen layer 19 is computed once per window and cached (float16). Before training, a check
  must show that running layers 20-23 and the final layer norm on the cache reproduces the saved FV1 WLM embeddings
  (max absolute difference below 1e-2 on float16 cache; cosine above 0.9999).
- Head: mean over frames -> dropout 0.1 -> linear 1024 -> 1 (logit). Dropout inside trained layers follows the checkpoint
  config (hidden dropout 0.1); layer drop is disabled.

Training
- Loss: binary cross-entropy with logits, one label per window (the person's label).
- Batch sampling: each mini-batch has 8 windows: 4 from people with PD and 4 from controls. Within a class a person is drawn
  uniformly, then one of that person's windows uniformly. So each class and each person has equal expected weight.
- One epoch = ceil(number of training windows / 8) steps. At most 6 epochs.
- Optimiser: AdamW, weight decay 0.01, learning rate lr for the trained transformer layers and layer norm, 1e-3 for the head;
  linear warm-up over the first 10% of the steps of epoch 1, then constant; gradient-norm clip 1.0; bf16 autocast on GPU.
- Seeds: torch/numpy seed 42 + 10 * outer fold + inner fold.

Search space (listed order = tie order): k in {2, 4} x lr in {1e-5, 3e-5} = 4 configurations.

Selection and refit
- For every configuration and every epoch, the three inner validation folds' person scores are pooled and the speaker AUPRC
  is computed. The (configuration, epoch) pair with the highest value is chosen; ties go to the earlier configuration, then
  the fewer epochs.
- The chosen configuration is refitted on all outer-training people for the chosen number of epochs (no early stopping in
  the refit) and predicts the outer test fold.
- Recording probability = mean of its window probabilities; person score = mean of recording probabilities.

Reported: nested held-out metrics (same metrics as FINAL_V1), learning curves (training loss, inner validation AUPRC per
epoch), training-partition scores of the refit, trainable parameter counts, and runtime.

## Decision rule for "the extension improves on the baseline" (added 22 October 2026, before any S12 or S13 outcome existed)

The baseline is FV1-WLM-ALL-LR (frozen WavLM vectors + logistic regression, official FINAL_V1 run). An extension
(S11 Random Forest, FV1-WLM-ALL-XGB, the S12 wide-grid nested models, the S13 fine-tuned model) counts as an
improvement only if both hold on the same 47 people:
1. it makes at least two fewer person-level errors (FN + FP at threshold 0.5) than the baseline, and
2. the 95% paired person-bootstrap interval (2,000 resamples, seed 42) of its AUROC minus the baseline AUROC excludes zero.
Otherwise the result is reported as "no improvement shown", with the reasons analysed.
