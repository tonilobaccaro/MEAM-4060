"""
Logistic regression flood-risk classifier.

Inputs  : outputs/X_scaled.npy, outputs/y_labels.npy, outputs/feature_names.txt
Outputs : outputs/lr_model.pkl, outputs/risk_scores.npy,
          outputs/feature_importance.png, outputs/risk_score_distribution.png
"""

import pathlib
import sys

import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
X_PATH      = pathlib.Path("outputs/X_scaled.npy")
Y_PATH      = pathlib.Path("outputs/y_labels.npy")
NAMES_PATH  = pathlib.Path("outputs/feature_names.txt")
MODEL_OUT   = pathlib.Path("outputs/lr_model.pkl")
SCORES_OUT  = pathlib.Path("outputs/risk_scores.npy")
IMP_OUT     = pathlib.Path("outputs/feature_importance.png")
HIST_OUT    = pathlib.Path("outputs/risk_score_distribution.png")

THRESHOLDS  = (33, 67)           # low/medium and medium/high boundaries

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
for p in (X_PATH, Y_PATH, NAMES_PATH):
    if not p.exists():
        print(f"ERROR: {p} not found — run feature_engineering.py first.")
        sys.exit(1)

X      = np.load(str(X_PATH))
y      = np.load(str(Y_PATH))
names  = NAMES_PATH.read_text().strip().splitlines()

print(f"Loaded  X: {X.shape}   y: {y.shape}   features: {len(names)}")
print(f"Class distribution — y=1: {y.sum():,}  y=0: {(y==0).sum():,}\n")

# ---------------------------------------------------------------------------
# 2. 5-fold stratified cross-validation
# ---------------------------------------------------------------------------
lr_params = dict(C=1.0, max_iter=2000, solver="lbfgs", random_state=42)

cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_acc  = []
cv_auc  = []
cv_f1   = []

print("5-fold stratified cross-validation …")
for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    clf = LogisticRegression(**lr_params)
    clf.fit(X_tr, y_tr)

    y_pred  = clf.predict(X_val)
    y_prob  = clf.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_prob) if len(np.unique(y_val)) > 1 else float("nan")
    f1  = f1_score(y_val, y_pred, zero_division=0)

    cv_acc.append(acc)
    cv_auc.append(auc)
    cv_f1.append(f1)

    print(f"  Fold {fold}:  acc={acc:.4f}  auc={auc:.4f}  f1={f1:.4f}")

print()
print(f"  Mean ± Std")
print(f"  Accuracy : {np.mean(cv_acc):.4f} ± {np.std(cv_acc):.4f}")
print(f"  ROC-AUC  : {np.nanmean(cv_auc):.4f} ± {np.nanstd(cv_auc):.4f}")
print(f"  F1       : {np.mean(cv_f1):.4f} ± {np.std(cv_f1):.4f}")
print()

# ---------------------------------------------------------------------------
# 3. Final model on all data
# ---------------------------------------------------------------------------
print("Training final model on full dataset …")
model = LogisticRegression(**lr_params)
model.fit(X, y)

joblib.dump(model, str(MODEL_OUT))
print(f"Model saved → {MODEL_OUT}\n")

# ---------------------------------------------------------------------------
# 4. Risk scores (0-100)
# ---------------------------------------------------------------------------
risk_scores = model.predict_proba(X)[:, 1] * 100

np.save(str(SCORES_OUT), risk_scores)
print(f"Risk scores saved → {SCORES_OUT}")
print(f"  min={risk_scores.min():.1f}  max={risk_scores.max():.1f}  "
      f"mean={risk_scores.mean():.1f}  median={np.median(risk_scores):.1f}\n")

# ---------------------------------------------------------------------------
# 5. Feature importance bar chart
# ---------------------------------------------------------------------------
coefs  = model.coef_[0]                          # shape (n_features,)
order  = np.argsort(np.abs(coefs))               # ascending abs value
sorted_names = [names[i] for i in order]
sorted_coefs = coefs[order]
colors = ["#F44336" if c > 0 else "#2196F3" for c in sorted_coefs]

fig, ax = plt.subplots(figsize=(9, max(4, 0.5 * len(names) + 1.5)))
bars = ax.barh(sorted_names, sorted_coefs, color=colors, edgecolor="white",
               height=0.6)

# Annotate bars with coefficient value
for bar, val in zip(bars, sorted_coefs):
    x_pos = val + 0.02 * np.sign(val) if val != 0 else 0.02
    ha    = "left" if val >= 0 else "right"
    ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}", va="center", ha=ha, fontsize=8.5)

ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Standardized Coefficient", fontsize=11)
ax.set_title("Logistic Regression Feature Importances (Standardized Coefficients)",
             fontsize=11, fontweight="bold")

red_patch  = mpatches.Patch(color="#F44336", label="Increases risk")
blue_patch = mpatches.Patch(color="#2196F3", label="Decreases risk")
ax.legend(handles=[red_patch, blue_patch], fontsize=9, loc="lower right")

fig.tight_layout()
fig.savefig(str(IMP_OUT), dpi=200)
print(f"Feature importance plot → {IMP_OUT}")

# ---------------------------------------------------------------------------
# 6. Plain-English coefficient interpretations
# ---------------------------------------------------------------------------
MAGNITUDE = {          # thresholds for magnitude language
    2.0: "very strongly",
    1.0: "substantially",
    0.5: "moderately",
    0.0: "slightly",
}

def magnitude_word(val: float) -> str:
    av = abs(val)
    for threshold, word in MAGNITUDE.items():
        if av >= threshold:
            return word
    return "negligibly"

DIRECTION = {
    "elev_ft_navd88":    ("higher ground elevation", "lower"),
    "bfe_ft":            ("higher BFE", "higher"),
    "freeboard_ft":      ("more freeboard (higher above BFE)", "lower"),
    "d_100yr_2025":      ("greater current 100-yr flood depth", "higher"),
    "d_100yr_2050_int":  ("greater projected 2050 flood depth", "higher"),
    "fld_zone_risk":     ("higher-risk flood zone (e.g. VE/AE)", "higher"),
    "dist_to_coast_m":   ("greater distance from Biscayne Bay", "lower"),
}

print("\n--- Coefficient Interpretations ---")
for name, coef in sorted(zip(names, coefs), key=lambda x: -abs(x[1])):
    direction = "increase" if coef > 0 else "decrease"
    mag       = magnitude_word(coef)
    feat_desc, risk_dir = DIRECTION.get(name, (name, direction + "d"))
    print(
        f"  {name} (coef={coef:+.3f}): "
        f"parcels with {feat_desc} have {mag} {risk_dir} predicted flood risk."
    )
print()

# ---------------------------------------------------------------------------
# 8. Risk score histogram with coloured bins
# ---------------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(9, 5))

bins = np.linspace(0, 100, 21)       # 20 equal-width bins
bin_centers = (bins[:-1] + bins[1:]) / 2

counts, _ = np.histogram(risk_scores, bins=bins)
bar_colors = [
    "#4CAF50" if c < THRESHOLDS[0] else "#FFC107" if c < THRESHOLDS[1] else "#F44336"
    for c in bin_centers
]

ax2.bar(bin_centers, counts, width=5.0, color=bar_colors, edgecolor="white",
        linewidth=0.5, align="center")

for thr, label in zip(THRESHOLDS, ["Low / Medium", "Medium / High"]):
    ax2.axvline(thr, color="black", linestyle="--", linewidth=1.1)
    ax2.text(thr + 0.8, ax2.get_ylim()[1] * 0.97, label,
             fontsize=8, va="top", color="black")

green_p  = mpatches.Patch(color="#4CAF50",  label="Low risk (< 33)")
yellow_p = mpatches.Patch(color="#FFC107",  label="Medium risk (33–67)")
red_p    = mpatches.Patch(color="#F44336",  label="High risk (> 67)")
ax2.legend(handles=[green_p, yellow_p, red_p], fontsize=9, loc="upper center")

ax2.set_xlabel("Risk Score (0 = lowest, 100 = highest)", fontsize=11)
ax2.set_ylabel("Number of Parcels", fontsize=11)
ax2.set_title("Flood Risk Score Distribution — Brickell Study Area",
              fontsize=12, fontweight="bold")
ax2.set_xlim(0, 100)
ax2.grid(axis="y", alpha=0.3)

fig2.tight_layout()
fig2.savefig(str(HIST_OUT), dpi=200)
print(f"Risk score histogram → {HIST_OUT}")
