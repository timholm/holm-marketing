"""Explore alternative weight configurations on the saved validation data."""

import numpy as np
import json
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix

data = np.load("/tmp/weights_validation_data.npz", allow_pickle=True)
X = data["X"]
y = data["y"]
print(f"loaded X={X.shape} y={y.shape} (pos={int(y.sum())} neg={int((1-y).sum())})")

# Distribution stats
print("\n=== component stats ===")
for i, name in enumerate(["mean_prob", "centroid_sim", "topic_bonus"]):
    print(f"  {name}: pos_mean={X[y==1,i].mean():.3f} neg_mean={X[y==0,i].mean():.3f} all_std={X[:,i].std():.3f}")
    # Univariate AUC
    print(f"    univariate AUC = {roc_auc_score(y, X[:,i]):.4f}")

# Try several weight schemes
schemes = [
    ("current", [0.55, 0.30, 0.15]),
    ("drop_centroid_renorm", [0.65, 0.0, 0.35]),
    ("drop_centroid_50_50", [0.50, 0.0, 0.50]),
    ("drop_centroid_60_40", [0.60, 0.0, 0.40]),
    ("drop_centroid_30_70", [0.30, 0.0, 0.70]),
    ("topic_dominant", [0.20, 0.0, 0.80]),
    ("equal_no_centroid", [0.50, 0.0, 0.50]),
    ("only_topic", [0.0, 0.0, 1.0]),
    ("only_meanprob", [1.0, 0.0, 0.0]),
    ("lr_normalized", [0.1865, 0.0, 0.8135]),
    ("lr_normalized_with_centroid", [0.30, 0.10, 0.60]),  # gut tweak
]

print("\n=== scheme comparison ===")
print(f"{'scheme':<32} {'AUC':>7} {'P@0.5':>7} {'R@0.5':>7} {'FPR@0.5':>8} {'mean(pos)':>10} {'mean(neg)':>10} {'n_above_0.5':>11}")
for label, w in schemes:
    w = np.array(w)
    scores = X @ w
    auc = roc_auc_score(y, scores)
    yhat = (scores >= 0.5).astype(int)
    p = precision_score(y, yhat, zero_division=0)
    r = recall_score(y, yhat, zero_division=0)
    cm = confusion_matrix(y, yhat).ravel()
    if cm.size == 4:
        tn, fp, fn_, tp = cm
        fpr = fp / max(fp + tn, 1)
    else:
        fpr = float("nan")
    n_above = int((scores >= 0.5).sum())
    print(f"{label:<32} {auc:>7.4f} {p:>7.3f} {r:>7.3f} {fpr:>8.3f} {scores[y==1].mean():>10.3f} {scores[y==0].mean():>10.3f} {n_above:>11d}")

# Find best non-negative weights via grid + pick best on AUC where threshold > 0.5 still has reasonable recall
print("\n=== grid search over (w1, w2, w3) where w_i in {0.0..1.0 in 0.05 steps}, sum=1 ===")
best = None
all_results = []
for w1 in np.arange(0.0, 1.01, 0.05):
    for w2 in np.arange(0.0, 1.01 - w1, 0.05):
        w3 = 1.0 - w1 - w2
        if w3 < 0:
            continue
        w = np.array([w1, w2, w3])
        scores = X @ w
        auc = roc_auc_score(y, scores)
        yhat = (scores >= 0.5).astype(int)
        # require recall@0.5 to be at least 0.20 so we don't pick a weight
        # config that scores nothing above 0.5
        r = recall_score(y, yhat, zero_division=0)
        if r < 0.15:
            continue
        all_results.append((auc, r, w.tolist()))
        if best is None or auc > best[0]:
            best = (auc, r, w.tolist())

print(f"  best: AUC={best[0]:.4f}, recall@0.5={best[1]:.3f}, w={best[2]}")
all_results.sort(reverse=True)
print("  top 10:")
for auc, r, w in all_results[:10]:
    print(f"    AUC={auc:.4f} recall={r:.3f} w={w}")

# AUC-only ranking (no recall constraint), broader grid
print("\n=== pure AUC grid (no recall filter) ===")
best_auc = None
for w1 in np.arange(0.0, 1.01, 0.05):
    for w2 in np.arange(0.0, 1.01 - w1, 0.05):
        w3 = 1.0 - w1 - w2
        if w3 < 0: continue
        w = np.array([w1, w2, w3])
        auc = roc_auc_score(y, X @ w)
        if best_auc is None or auc > best_auc[0]:
            best_auc = (auc, w.tolist())
print(f"  pure best: AUC={best_auc[0]:.4f}, w={best_auc[1]}")

# Now: optimize for AUC while keeping score MAGNITUDE comparable to current.
# Current pos mean = 0.39, current spans [0.13, 0.78].
# We want the new scheme to keep mean(score) ~ 0.30 so the existing 0.5
# threshold remains meaningful.
print("\n=== AUC max with score-mean constrained near 0.30 ===")
best_constrained = None
for w1 in np.arange(0.0, 1.01, 0.05):
    for w2 in np.arange(0.0, 1.01 - w1, 0.05):
        w3 = 1.0 - w1 - w2
        if w3 < 0: continue
        w = np.array([w1, w2, w3])
        scores = X @ w
        m = scores.mean()
        # Allow mean in [0.20, 0.40] so threshold 0.5 still gates
        if not (0.20 <= m <= 0.40):
            continue
        auc = roc_auc_score(y, scores)
        if best_constrained is None or auc > best_constrained[0]:
            best_constrained = (auc, m, w.tolist())
print(f"  constrained best: AUC={best_constrained[0]:.4f}, mean(score)={best_constrained[1]:.3f}, w={best_constrained[2]}")

# Final verdict
print("\n=== final picks ===")
for label, w in [
    ("current (baseline)", [0.55, 0.30, 0.15]),
    ("lr_normalized", [0.1865, 0.0, 0.8135]),
    ("constrained_best", best_constrained[2]),
]:
    w = np.array(w)
    scores = X @ w
    auc = roc_auc_score(y, scores)
    print(f"  {label}: AUC={auc:.4f}, mean(pos)={scores[y==1].mean():.3f}, mean(neg)={scores[y==0].mean():.3f}, n_above_0.5={(scores>=0.5).sum()}")

