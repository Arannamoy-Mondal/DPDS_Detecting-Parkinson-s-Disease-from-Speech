# ENVIRONMENT (recorded 2026-10-20T09:37:09Z)
- OS: Linux (Linux-7.0.0-38-generic-x86_64-with-glibc2.43), kernel 7.0.0-38-generic
- CPU: Intel(R) Core(TM) i9-14900K, logical cores: 32
- RAM: 61 GB total
- Disk (workspace): `/dev/nvme0n1p7  679G  554G   91G  86% /root/Projects`
- GPU/accelerator: none used. All computation runs on CPU (torch.cuda.is_available() is not relied on; `CUDA_VISIBLE_DEVICES=""` is set for extraction).
- Python: 3.11.17 (virtual environment `/root/Projects/Project4a_FINAL_V1_ISOLATED/.venv`, created with uv)
- Main packages:
```
huggingface-hub==2.2.0
joblib==1.6.0
numpy==2.4.6
pandas==3.0.6
scikit-learn==1.9.1
scipy==1.17.1
soundfile==0.14.0
torch==2.14.1
transformers==5.20.2
xgboost==3.2.0
```
- Hugging Face cache: `cache/hf` inside the workspace (HF_HOME), offline mode during extraction.
- Nondeterminism limits (updated by the audit, 20 October 2026): re-extracting embeddings in a fresh process reproduces the saved vectors to within 3.1e-6 (cosine 1.0). Model fitting uses the saved arrays and fixed seeds. Re-executing the 70 logistic-regression runs with single-threaded BLAS reproduced the official results bit-identically; with unrestricted multi-threaded BLAS, L2-logistic regression (liblinear trust-region solver, which uses BLAS dot products) can move probabilities by up to 4e-4 without changing decisions. XGBoost uses `hist` with fixed `random_state`.

## Package additions after the first recording of this file
- `torch` is the `2.14.1+cu130` build (a CUDA build; only the CPU was used).
- Installed versions (`uv pip freeze`): matplotlib==3.11.2 openpyxl==3.1.5 threadpoolctl==3.7.0 torch==2.14.1 
- `openpyxl` was added during the audit only to read the corpus spreadsheets (demographics); `matplotlib` is used for report figures only.
