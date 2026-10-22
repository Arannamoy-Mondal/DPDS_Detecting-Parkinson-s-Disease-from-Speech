"""S13 step 2 (amendment A3, exploratory): nested partial fine-tuning of WavLM-large on the FINAL_V1 folds.

Settings are fixed in supplementary/PREDECLARED_A3.md. Starts from the cache of frozen layers 0-19 (s13_cache.py, check
must have passed). Trains the top k transformer layers, the final layer norm and a linear head. Per outer fold:
4 configurations x 3 inner folds x up to 6 epochs, selection by pooled inner speaker AUPRC, then a refit on all
outer-training people and prediction of the outer test fold. Each finished outer fold is saved and skipped on restart.
"""
import os, sys, json, time, copy, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from supp_common import *
os.environ["HF_HOME"] = str(ROOT / "cache/hf")
os.environ["HF_HUB_OFFLINE"] = "1"
import torch
import torch.nn as nn
from transformers import AutoModel
from splits import assign_folds

CACHE = ROOT / "cache/ft_wlm_L20"
OUTD = RES / "S13_folds"
CONFIGS = [{"k": 2, "lr": 1e-5}, {"k": 2, "lr": 3e-5}, {"k": 4, "lr": 1e-5}, {"k": 4, "lr": 3e-5}]
MAX_EPOCHS = 6
BATCH = 8
HEAD_LR = 1e-3
DEVICE = "cuda"
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class FineTuned(nn.Module):
    """Frozen layers 20..23-k (no gradient), trainable layers 24-k..23, final layer norm and a linear head."""

    def __init__(self, base, k):
        super().__init__()
        self.frozen = nn.ModuleList([base.encoder.layers[i] for i in range(20, 24 - k)])
        self.trained = nn.ModuleList([copy.deepcopy(base.encoder.layers[i]) for i in range(24 - k, 24)])
        self.norm = copy.deepcopy(base.encoder.layer_norm)
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(1024, 1))
        self.att0 = base.encoder.layers[0].attention
        self.bias_cache = {}
        for p in self.frozen.parameters():
            p.requires_grad_(False)
        # deepcopy keeps requires_grad=False from the frozen base model; the copied layers must be trainable
        for p in list(self.trained.parameters()) + list(self.norm.parameters()):
            p.requires_grad_(True)

    def raw_bias(self, T, bsz):
        if T not in self.bias_cache:
            with torch.no_grad():
                self.bias_cache[T] = self.att0.compute_bias(T, T)
        b = self.bias_cache[T]
        return b.unsqueeze(0).expand(bsz, -1, -1, -1).reshape(bsz * self.att0.num_heads, T, T)

    def forward(self, h):
        h = h.float()
        pb = self.raw_bias(h.shape[1], h.shape[0])
        with torch.no_grad():
            for layer in self.frozen:
                layer.eval()
                h, pb = layer(h, position_bias=pb)
        for layer in self.trained:
            h, pb = layer(h, position_bias=pb)
        return self.head(self.norm(h).mean(dim=1)).squeeze(-1)


def trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Data:
    def __init__(self):
        self.idx = pd.read_csv(CACHE / "window_index.csv")
        arr = np.load(CACHE / "hidden_L20_fp16.npy", mmap_mode="r")
        self.h = torch.from_numpy(np.ascontiguousarray(arr)).to(DEVICE)
        self.by_person = {s: g.index.values for s, g in self.idx.groupby("speaker_id")}
        self.label_of = self.idx.drop_duplicates("speaker_id").set_index("speaker_id").label.to_dict()
        self.start = self.idx.start.values
        self.frames = self.idx.frames.values

    def window(self, i):
        return self.h[self.start[i]:self.start[i] + self.frames[i]]

    def batches(self, rows, size=32):
        """Group window rows by length so that each forward pass stacks equal-length windows."""
        frames = self.idx.frames.values[rows]
        for T in np.unique(frames):
            sel = rows[frames == T]
            for j in range(0, len(sel), size):
                part = sel[j:j + size]
                yield part, torch.stack([self.window(i) for i in part])


def predict(model, data, people):
    model.eval()
    rows = np.concatenate([data.by_person[s] for s in people])
    out = np.empty(len(rows))
    pos = {r: n for n, r in enumerate(rows)}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for part, x in data.batches(rows):
            p = torch.sigmoid(model(x).float()).cpu().numpy()
            for i, v in zip(part, p):
                out[pos[i]] = v
    w = data.idx.iloc[rows][["sample_id", "speaker_id", "label"]].copy()
    w["p"] = out
    rec = w.groupby("sample_id").agg(p=("p", "mean"), speaker_id=("speaker_id", "first"), label=("label", "first"))
    return rec


def window_loss(model, data, people):
    model.eval()
    rows = np.concatenate([data.by_person[s] for s in people])
    total, n = 0.0, 0
    lossf = nn.BCEWithLogitsLoss(reduction="sum")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for part, x in data.batches(rows):
            y = torch.tensor(data.idx.label.values[part], dtype=torch.float32, device=DEVICE)
            total += float(lossf(model(x).float(), y)); n += len(part)
    return total / n


def train(base, data, train_people, cfg, epochs, seed, eval_people=None):
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    model = FineTuned(base, cfg["k"]).to(DEVICE)
    enc_params = list(model.trained.parameters()) + list(model.norm.parameters())
    expected = sum(p.numel() for p in enc_params) + sum(p.numel() for p in model.head.parameters())
    if trainable_parameters(model) != expected or expected < 1_000_000:
        print("STOP: trainable parameters %d, expected %d" % (trainable_parameters(model), expected)); sys.exit(1)
    start_weights = [p.detach().clone() for p in model.trained.parameters()]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": cfg["lr"]}, {"params": model.head.parameters(), "lr": HEAD_LR}], weight_decay=0.01)
    pd_people = [s for s in train_people if data.label_of[s] == 1]
    hc_people = [s for s in train_people if data.label_of[s] == 0]
    n_windows = sum(len(data.by_person[s]) for s in train_people)
    steps = math.ceil(n_windows / BATCH)
    warm = max(1, int(0.1 * steps))
    lossf = nn.BCEWithLogitsLoss()
    curve, step_no = [], 0
    for epoch in range(1, epochs + 1):
        model.train()
        for layer in model.frozen:
            layer.eval()
        losses = []
        for _ in range(steps):
            rows = []
            for group in (pd_people, hc_people):
                for s in rng.choice(group, BATCH // 2):
                    rows.append(rng.choice(data.by_person[s]))
            rows = np.array(rows)
            y = torch.tensor(data.idx.label.values[rows], dtype=torch.float32, device=DEVICE)
            scale = min(1.0, (step_no + 1) / warm) if epoch == 1 else 1.0
            for g, base_lr in zip(opt.param_groups, [cfg["lr"], HEAD_LR]):
                g["lr"] = base_lr * scale
            opt.zero_grad(set_to_none=True)
            loss_sum = 0.0
            frames = data.idx.frames.values[rows]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for T in np.unique(frames):
                    sel = np.where(frames == T)[0]
                    x = torch.stack([data.window(i) for i in rows[sel]])
                    l = lossf(model(x).float(), y[sel]) * len(sel) / len(rows)
                    l.backward()
                    loss_sum += float(l)
            torch.nn.utils.clip_grad_norm_(enc_params + list(model.head.parameters()), 1.0)
            opt.step()
            losses.append(loss_sum); step_no += 1
        entry = {"epoch": epoch, "train_loss": float(np.mean(losses))}
        if eval_people is not None:
            rec = predict(model, data, eval_people)
            entry["eval_rec"] = rec
            entry["eval_loss"] = window_loss(model, data, eval_people)
        entry["max_weight_change"] = max(float((p.detach() - q).abs().max()) for p, q in zip(model.trained.parameters(), start_weights))
        curve.append(entry)
    if curve[-1]["max_weight_change"] == 0.0:
        print("STOP: the trained WavLM layers did not change"); sys.exit(1)
    return model, curve


def speaker_auprc_rec(rec):
    g = rec.groupby("speaker_id").agg(s=("p", "mean"), y=("label", "first"))
    return float(pl.speaker_auprc(rec.p.values, rec.speaker_id.values, rec.label.values.astype(int))), g


def run_fold(base, data, fold, people_fold):
    t0 = time.time()
    test_people = sorted(s for s, f in people_fold.items() if f == fold)
    train_people = sorted(s for s, f in people_fold.items() if f != fold)
    labels = {s: data.label_of[s] for s in train_people}
    inner_map = assign_folds(train_people, labels, P.N_INNER, 142 + fold)
    curves = []
    pooled = {}
    for ci, cfg in enumerate(CONFIGS):
        per_epoch = {e: [] for e in range(1, MAX_EPOCHS + 1)}
        for k in range(P.N_INNER):
            tr = [s for s in train_people if inner_map[s] != k]
            va = [s for s in train_people if inner_map[s] == k]
            _, curve = train(base, data, tr, cfg, MAX_EPOCHS, 42 + 10 * fold + k, eval_people=va)
            for c in curve:
                per_epoch[c["epoch"]].append(c["eval_rec"])
                curves.append({"fold": fold, "config_index": ci, "k": cfg["k"], "lr": cfg["lr"], "inner_fold": k, "epoch": c["epoch"],
                               "train_loss": c["train_loss"], "val_loss": c["eval_loss"], "val_auprc_this_inner_fold": speaker_auprc_rec(c["eval_rec"])[0],
                               "max_weight_change": c["max_weight_change"]})
            torch.cuda.empty_cache()
        for e, recs in per_epoch.items():
            pooled[(ci, e)] = speaker_auprc_rec(pd.concat(recs))[0]
        print("fold", fold, "config", cfg, "inner AUPRC by epoch", [round(pooled[(ci, e)], 4) for e in range(1, MAX_EPOCHS + 1)],
              "%.0f s" % (time.time() - t0), flush=True)
    best = max(pooled, key=lambda ce: (round(pooled[ce], 12), -ce[0], -ce[1]))
    ci, ep = best
    model, refit_curve = train(base, data, train_people, CONFIGS[ci], ep, 42 + 10 * fold + 9)
    test_rec = predict(model, data, test_people)
    train_rec = predict(model, data, train_people)
    out = {"fold": fold, "selected": CONFIGS[ci], "config_index": ci, "epochs": ep, "inner_auprc_selected": pooled[best],
           "inner_auprc_table": [{"config_index": c, "epoch": e, "pooled_inner_auprc": v} for (c, e), v in sorted(pooled.items())],
           "refit_train_loss": [c["train_loss"] for c in refit_curve], "trainable_parameters": trainable_parameters(model),
           "refit_max_weight_change": refit_curve[-1]["max_weight_change"],
           "runtime_seconds": time.time() - t0}
    OUTD.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(curves).to_csv(OUTD / ("fold%d_curves.csv" % fold), index=False)
    t = test_rec.reset_index(); t["outer_fold"] = fold; t.to_csv(OUTD / ("fold%d_test.csv" % fold), index=False)
    t = train_rec.reset_index(); t["outer_fold"] = fold; t.to_csv(OUTD / ("fold%d_train.csv" % fold), index=False)
    json.dump(out, open(OUTD / ("fold%d.json" % fold), "w"), indent=2)
    print("fold", fold, "selected", CONFIGS[ci], "epochs", ep, "inner %.4f" % pooled[best], "%.0f s" % (time.time() - t0), flush=True)
    del model
    torch.cuda.empty_cache()


def main():
    summ = json.loads((RES / "S13_cache_summary.json").read_text())
    if not summ["check_passed"]:
        print("STOP: cache check failed; no training."); sys.exit(1)
    ck = json.loads((ROOT / "features/manifests/encoder_checkpoints.json").read_text())["WLM"]
    base = AutoModel.from_pretrained(ck["snapshot"], dtype=torch.float32).eval()
    base.encoder.layers[0].to(DEVICE)
    for i in range(20, 24):
        base.encoder.layers[i].to(DEVICE)
    base.encoder.layer_norm.to(DEVICE)
    for p in base.parameters():
        p.requires_grad_(False)
    data = Data()
    split = pd.read_csv(ROOT / "SPLIT_MANIFEST.csv").drop_duplicates("speaker_id").set_index("speaker_id").outer_fold.to_dict()
    folds = [int(a) for a in sys.argv[1:]] or list(range(5))
    for fold in folds:
        if (OUTD / ("fold%d.json" % fold)).exists():
            print("fold", fold, "already done; skipped"); continue
        run_fold(base, data, fold, split)
    # pooled summary once all five folds exist
    if all((OUTD / ("fold%d.json" % f)).exists() for f in range(5)):
        test = pd.concat([pd.read_csv(OUTD / ("fold%d_test.csv" % f)) for f in range(5)])
        g = test.groupby("speaker_id").agg(s=("p", "mean"), y=("label", "first"), fold=("outer_fold", "first"))
        m = pl.binary_metrics(g.s.values, g.y.values.astype(int))
        pred = test.rename(columns={"p": "predicted_probability", "label": "true_label"})[["sample_id", "speaker_id", "outer_fold", "true_label", "predicted_probability"]]
        pred.to_csv(RES / "S13_ft_sample_predictions.csv", index=False)
        folds_info = [json.loads((OUTD / ("fold%d.json" % f)).read_text()) for f in range(5)]
        json.dump({"pooled": m, "folds": folds_info}, open(RES / "S13_ft_summary.json", "w"), indent=2, default=float)
        print("pooled fine-tuned:", {k: round(m[k], 4) for k in ["f1", "sensitivity", "specificity", "auroc", "auprc"]})


if __name__ == "__main__":
    main()
