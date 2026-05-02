"""
train_model_v6.py — Binary critical-vs-non-critical classifier
================================================================

WHY BINARY (changes vs train_model_v5)
--------------------------------------
The MIMIC demo distribution is 4 / 43 / 20 / 3 / 0 patients across ESI 1-5.
After subject-isolated split, ESI-1 has 3 train patients, ESI-4 has 2, and
ESI-5 has zero real patients. No multi-class model can honestly learn from
that — augmentation just makes copies that the model memorizes, which is
why train_model_v5 collapses to predicting ESI-2 for everything (87.5%
recall on ESI-2, 0% on every other class).

The fix is to reframe the task: the model decides ONLY "is this patient
critical?" (ESI 1-2 vs ESI 3-5). Real distribution becomes 47 critical /
23 non-critical, which is feasible for honest training.

The granular ESI 1-5 score is then produced downstream by the clinical
override layer in api/main.py based on vital-sign rules — not by the model.
This matches how real clinical decision-support systems work: ML provides
a prior, rules provide the safety floor.

INPUTS  (run pipeline_v3.py first to produce these)
  mimic-iii-clinical-db/mimic-iii-train.csv
  mimic-iii-clinical-db/mimic-iii-real.csv

OUTPUTS  (saved with _binary suffix so v5 artifacts are preserved)
  models/triage_model_binary.pkl       — calibrated binary classifier
  models/feature_names_binary.pkl      — feature list (same set as v5)
  models/real_median_binary.pkl        — for NaN imputation at inference
  models/train_median_binary.pkl       — diagnostic
  models/evaluation_report_binary.txt  — F1 + safety recall metrics
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.utils.class_weight import compute_sample_weight

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
Path("models").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)


# ─── FEATURES (identical set to train_model_v5) ─────────────────────────
BASE_FEATURES = [
    "heart_rate", "systolic_bp", "diastolic_bp",
    "resp_rate", "spo2", "temperature",
    "age", "gender_m", "complaint_cat",
    "has_diabetes", "has_hypertension", "has_heart_disease",
]


# ─── LOAD ───────────────────────────────────────────────────────────────
df_real  = pd.read_csv("mimic-iii-clinical-db/mimic-iii-real.csv")
df_train = pd.read_csv("mimic-iii-clinical-db/mimic-iii-train.csv")

EMB_FEATURES = [c for c in df_real.columns if c.startswith("complaint_emb_")]
FEATURE_COLS = [
    f for f in BASE_FEATURES + EMB_FEATURES
    if f in df_real.columns and f in df_train.columns
]

print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
print(f"\nReal MIMIC rows (test) : {len(df_real)}")
print(f"Training rows           : {len(df_train)}")


# ─── BINARY LABEL ───────────────────────────────────────────────────────
# 1 = critical (ESI 1 or 2), 0 = non-critical (ESI 3, 4, or 5)
y_train_binary = (df_train["esi_level"] <= 2).astype(int)
y_test_binary  = (df_real["esi_level"]  <= 2).astype(int)

print("\nTrain class distribution (binary):")
print(f"  non_critical (ESI 3-5): {(y_train_binary == 0).sum()}")
print(f"  critical     (ESI 1-2): {(y_train_binary == 1).sum()}")

print("\nTest class distribution (binary):")
print(f"  non_critical (ESI 3-5): {(y_test_binary == 0).sum()}")
print(f"  critical     (ESI 1-2): {(y_test_binary == 1).sum()}")


# ─── PREPARE FEATURE MATRICES ───────────────────────────────────────────
train_median = df_train[FEATURE_COLS].median()
real_median  = df_real[FEATURE_COLS].median()

X_train = df_train[FEATURE_COLS].fillna(train_median).astype(np.float32)
X_test  = df_real[FEATURE_COLS].fillna(real_median).astype(np.float32)
groups  = df_train["subject_id"].values


# ─── TRAIN BINARY XGBOOST ───────────────────────────────────────────────
print("\nTraining binary XGBoost...")
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.7,
    colsample_bytree=0.7,
    min_child_weight=5,
    reg_alpha=0.5,
    reg_lambda=2.0,
    gamma=0.3,
    eval_metric="logloss",
    objective="binary:logistic",
    random_state=42,
    n_jobs=-1,
)

sw_train = compute_sample_weight("balanced", y=y_train_binary).astype(np.float32)
model.fit(X_train, y_train_binary, sample_weight=sw_train, verbose=False)
print("Done.")


# ─── PROBABILITY CALIBRATION (sigmoid, held-out 20%) ────────────────────
print("\nCalibrating probabilities (sigmoid, held-out cal set)...")
X_fit, X_cal, y_fit, y_cal = train_test_split(
    X_train, y_train_binary,
    test_size=0.20,
    random_state=42,
    stratify=y_train_binary,
)
sw_fit = compute_sample_weight("balanced", y=y_fit).astype(np.float32)
model.fit(X_fit, y_fit, sample_weight=sw_fit, verbose=False)

calibrated = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
sw_cal = compute_sample_weight("balanced", y=y_cal).astype(np.float32)
calibrated.fit(X_cal, y_cal, sample_weight=sw_cal)
print("Calibration done.")


# ─── HONEST EVALUATION ON SUBJECT-ISOLATED TEST SET ─────────────────────
print("\n" + "=" * 60)
print("HONEST EVALUATION — BINARY (Critical vs Non-critical)")
print("=" * 60)

y_pred = calibrated.predict(X_test)
y_pred_proba = calibrated.predict_proba(X_test)[:, 1]    # probability of critical

weighted_f1     = f1_score(y_test_binary, y_pred, average="weighted", zero_division=0)
macro_f1        = f1_score(y_test_binary, y_pred, average="macro",    zero_division=0)
critical_recall = recall_score(y_test_binary, y_pred, pos_label=1, zero_division=0)
non_crit_recall = recall_score(y_test_binary, y_pred, pos_label=0, zero_division=0)

print(f"\nWeighted F1                : {weighted_f1:.4f}")
print(f"Macro F1                   : {macro_f1:.4f}")
print(f"Critical (ESI 1-2) Recall  : {critical_recall:.4f}  ← safety-critical")
print(f"Non-critical Recall        : {non_crit_recall:.4f}")

print("\nClassification report:")
print(classification_report(
    y_test_binary, y_pred,
    target_names=["non_critical", "critical"],
    zero_division=0,
))

cm = confusion_matrix(y_test_binary, y_pred)
print(f"Confusion matrix (rows=true, cols=pred):\n{cm}")

# Save confusion matrix plot
fig, ax = plt.subplots(figsize=(6, 5))
ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["non_critical", "critical"])
ax.set_yticklabels(["non_critical", "critical"])
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("Binary classifier — confusion matrix")
for (i, j), v in np.ndenumerate(cm):
    ax.text(j, i, str(v), ha="center", va="center", color="black", fontsize=14)
plt.tight_layout()
plt.savefig("reports/confusion_matrix_binary.png", dpi=150)
plt.close()
print("Confusion matrix saved → reports/confusion_matrix_binary.png")


# ─── STRATIFIED GROUP K-FOLD CV (subject_id as group) ───────────────────
print("\nStratifiedGroupKFold CV (subject_id groups)...")

# Synthetic rows in pipeline_v3 use subject_id <= 0; give them unique IDs so
# the splitter can distribute them across folds.
groups_for_cv = df_train["subject_id"].values.astype(int).copy()
synthetic_mask = groups_for_cv <= 0
if synthetic_mask.any():
    next_id = max(int(groups_for_cv.max()), 0) + 1
    groups_for_cv[synthetic_mask] = next_id + np.arange(synthetic_mask.sum())

cv_model = xgb.XGBClassifier(
    n_estimators=200, max_depth=3, learning_rate=0.05,
    subsample=0.7, colsample_bytree=0.7, min_child_weight=5,
    reg_alpha=0.5, reg_lambda=2.0, gamma=0.3,
    eval_metric="logloss", objective="binary:logistic",
    random_state=42, n_jobs=-1,
)

X_train_np = X_train.to_numpy()
y_train_np = y_train_binary.to_numpy()

cv_f1s              = []
cv_critical_recalls = []

sgkf = StratifiedGroupKFold(n_splits=5)
for fold, (tr_idx, val_idx) in enumerate(
        sgkf.split(X_train_np, y_train_np, groups=groups_for_cv)):
    if len(tr_idx) == 0 or len(val_idx) == 0:
        continue
    fold_w = compute_sample_weight("balanced", y=y_train_np[tr_idx]).astype(np.float32)
    cv_model.fit(X_train_np[tr_idx], y_train_np[tr_idx],
                 sample_weight=fold_w, verbose=False)
    val_y = y_train_np[val_idx]
    preds = cv_model.predict(X_train_np[val_idx])
    cv_f1s.append(f1_score(val_y, preds, average="weighted", zero_division=0))
    if (val_y == 1).any():
        cv_critical_recalls.append(
            recall_score(val_y, preds, pos_label=1, zero_division=0)
        )

cv_f1s_arr = np.array(cv_f1s) if cv_f1s else np.array([])
if len(cv_f1s_arr) > 0:
    print(f"CV F1 scores               : {cv_f1s_arr.round(4)}")
    print(f"Mean CV F1                 : {cv_f1s_arr.mean():.4f} ± {cv_f1s_arr.std():.4f}")
else:
    print("CV: no valid folds completed")

if cv_critical_recalls:
    print(f"CV critical recall (mean)  : {np.mean(cv_critical_recalls):.4f}")
print("\n(Train/Test F1 gap < 0.25 = trustworthy; > 0.40 = still leaking)")


# ─── SAVE ARTIFACTS ─────────────────────────────────────────────────────
print("\nSaving artifacts...")
with open("models/triage_model_binary.pkl",  "wb") as f: pickle.dump(calibrated, f)
with open("models/feature_names_binary.pkl", "wb") as f: pickle.dump(FEATURE_COLS, f)
with open("models/real_median_binary.pkl",   "wb") as f: pickle.dump(real_median.to_dict(), f)
with open("models/train_median_binary.pkl",  "wb") as f: pickle.dump(train_median.to_dict(), f)

cv_mean_str  = f"{cv_f1s_arr.mean():.4f} ± {cv_f1s_arr.std():.4f}" if len(cv_f1s_arr) > 0 else "N/A"
cv_crit_str  = f"{np.mean(cv_critical_recalls):.4f}" if cv_critical_recalls else "N/A"

eval_report = f"""
MODEL EVALUATION REPORT (BINARY)
==================================
Classifier task : critical (ESI 1-2) vs non-critical (ESI 3-5)
Features used   : {len(FEATURE_COLS)}

Test Performance (subject-isolated, real MIMIC)
-----------------------------------------------
Weighted F1                : {weighted_f1:.4f}
Macro F1                   : {macro_f1:.4f}
Critical (ESI 1-2) Recall  : {critical_recall:.4f}   ← safety-critical
Non-critical Recall        : {non_crit_recall:.4f}

Test class counts:
  non_critical  : {(y_test_binary == 0).sum()}
  critical      : {(y_test_binary == 1).sum()}

StratifiedGroupKFold CV (5 folds, subject_id groups)
----------------------------------------------------
CV F1 scores    : {cv_f1s_arr.round(4) if len(cv_f1s_arr) > 0 else 'N/A'}
Mean CV F1      : {cv_mean_str}
CV crit recall  : {cv_crit_str}

Note
----
This model produces a binary "is this patient critical?" decision.
Granular ESI 1-5 levels for the API response come from the clinical
override layer in api/main.py based on vital-sign rules, not from this
classifier. The override layer also catches dangerous cases the model
misses (severe hypoxia, hypotension, etc.) regardless of model output.
"""
with open("models/evaluation_report_binary.txt", "w", encoding="utf-8") as f:
    f.write(eval_report)

print("Saved: models/triage_model_binary.pkl (+ feature_names, real_median, train_median)")
print("Saved: models/evaluation_report_binary.txt")
print("\nDone. Restart uvicorn to pick up the new binary model.")
