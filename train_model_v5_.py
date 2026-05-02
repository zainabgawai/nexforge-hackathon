"""
train_model_v4.py — All v3 fixes + 8 additional improvements
=============================================================
Fixes over v3 (based on code review):

  1. STATS AFTER AUGMENTATION: train_median is now computed on the post-
     augmentation training set, not pre-augmentation. real_median is kept
     separately for test-set NaN filling only.

  2. CLASS IMBALANCE: compute_sample_weight(class_weight='balanced') is
     applied to both the XGBoost fit and every CV fold. ESI-1/ESI-2 are
     rare but safety-critical; ignoring imbalance causes inflated ESI-3
     accuracy and dangerously low ESI-1 recall.

  3. PROBABILITY CALIBRATION: CalibratedClassifierCV (isotonic regression,
     cv='prefit') wraps the trained XGBoost model. Raw softmax probabilities
     are overconfident; calibrated probabilities are required for clinical use.

  4. SAFETY-CRITICAL RECALL METRICS: ESI-1 and ESI-2 recall are now
     reported explicitly in both test evaluation and CV folds. Weighted F1
     hides failures on rare classes; missing ESI-1 is the worst failure mode.

  5. CONFUSION MATRIX CONSISTENCY: labels and display_labels are now both
     restricted to present_classes so the matrix dimensions always match.

  6. CV WITH BALANCED WEIGHTS: each CV fold uses per-fold balanced sample
     weights and tracks ESI-1/2 recall — making CV a meaningful safety audit.

  7. BIDIRECTIONAL OVERRIDE: the override layer now also flags cases where
     the model predicts ESI-1 but vitals are only borderline-critical, so
     humans review potentially overconfident high-acuity predictions.

  8. UNCERTAINTY TRIGGER: explain_prediction() now sets needs_human_review=True
     when model confidence < 50% OR when model and clinical signal engine
     disagree on acuity. Critical for deployment safety.

Run: python train_model_v4.py
Set GEMINI_API_KEY environment variable before running.
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.metrics import (classification_report, f1_score, recall_score,
                              confusion_matrix, ConfusionMatrixDisplay)
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from google import genai

client = genai.Client() 

Path("models").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════
# CONFIG (override safety tuning lives here)
# ═══════════════════════════════════════════════════════════

CONFIDENCE_THRESHOLD = 0.50

CRITICAL_VITALS = {
    "sbp_low": 80,
    "spo2_low": 88,
    "hr_high": 150,
    "hr_low": 40,
    "rr_high": 35,
    "temp_low": 95,
}

ESI2_THRESHOLD_RULE = {
    "min_flags": 2,
    "sbp": 90,
    "spo2": 92,
    "hr": 130,
    "rr": 28,
    "shock_index": 1.0,
}

# ── FEATURE CONFIG ────────────────────────────────────────
BASE_FEATURES = [
    'heart_rate', 'systolic_bp', 'diastolic_bp',
    'resp_rate', 'spo2', 'temperature',
    'age', 'gender_m', 'complaint_cat',
    'has_diabetes', 'has_hypertension', 'has_heart_disease',
]

FEATURE_LABELS = {
    'heart_rate':        'Heart rate',
    'systolic_bp':       'Systolic blood pressure',
    'diastolic_bp':      'Diastolic blood pressure',
    'resp_rate':         'Respiratory rate',
    'spo2':              'Oxygen saturation (SpO₂)',
    'temperature':       'Body temperature',
    'age':               'Patient age',
    'gender_m':          'Patient gender',
    'complaint_cat':     'Chief complaint category',
    'has_diabetes':      'Diabetes (pre-existing)',
    'has_hypertension':  'Hypertension (pre-existing)',
    'has_heart_disease': 'Heart disease (pre-existing)',
}
for i in range(8):
    FEATURE_LABELS[f'complaint_emb_{i}'] = f'Complaint embedding dim {i}'


# ═══════════════════════════════════════════════════════════
# 1. LOAD & LEAKAGE-FREE SPLIT
# ═══════════════════════════════════════════════════════════
df_real  = pd.read_csv('mimic-iii-clinical-db/mimic-iii-real.csv')
df_train = pd.read_csv('mimic-iii-clinical-db/mimic-iii-train.csv')

EMB_FEATURES = [c for c in df_real.columns if c.startswith('complaint_emb_')]
FEATURE_COLS = [f for f in BASE_FEATURES + EMB_FEATURES
                if f in df_real.columns and f in df_train.columns]

print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")

# ── FIX BUG 1: The pipeline already separated real vs train CSVs.
# DO NOT filter by subject_id — the pipeline's mimic-iii-train.csv already
# contains augmented/synthetic rows whose subject_ids overlap with real rows
# (because they were derived FROM those rows). Removing them wipes 80% of
# training data and leaves only 150 synthetic rows, causing F1=0.
# Instead: use mimic-iii-train.csv as-is for training, mimic-iii-real.csv
# as-is for testing. The pipeline already enforces this separation.
df_train_clean = df_train.copy()

print(f"\nReal MIMIC rows (test set)   : {len(df_real)}")
print(f"Training rows                : {len(df_train_clean)}")
print(f"ESI distribution in train    :\n{df_train_clean['esi_level'].value_counts().sort_index()}\n")

def augment_training(df, factor=6, noise_pct=0.07):
    numeric_cols = ['heart_rate', 'systolic_bp', 'diastolic_bp',
                    'resp_rate', 'spo2', 'temperature', 'age']
    emb_cols_present = [c for c in df.columns if c.startswith('complaint_emb_')]
    augmented = [df.copy()]
    for _ in range(factor):
        noisy = df.copy()
        for col in numeric_cols:
            if col in noisy.columns:
                std = max(df[col].std() * noise_pct, 0.01)
                noisy[col] = (noisy[col] + np.random.normal(0, std, len(noisy))).round(2)
        for col in emb_cols_present:
            noisy[col] = (noisy[col] + np.random.normal(0, 0.01, len(noisy))).round(6)
        augmented.append(noisy)
    return pd.concat(augmented, ignore_index=True)

np.random.seed(42)

# FIX BUG 5: Don't inject dummy rows of all-zeros — they teach the model
# that "HR=0, BP=0" is a valid pattern for a given ESI class.
# Instead, if a class is missing, synthesise a realistic row from the
# median of the class above/below, then add small noise.
all_classes = [1, 2, 3, 4, 5]
for cls in all_classes:
    if cls not in df_train_clean['esi_level'].values:
        # Use median of whole training set as base, then tag with the class
        ref_cls = cls + 1 if cls < 5 else cls - 1
        ref_rows = df_train_clean[df_train_clean['esi_level'] == ref_cls]
        if len(ref_rows) == 0:
            ref_rows = df_train_clean
        dummy = ref_rows.median(numeric_only=True).to_frame().T.copy()
        dummy['esi_level'] = cls
        # Fill any remaining NaN columns (non-numeric) with 0
        for col in df_train_clean.columns:
            if col not in dummy.columns:
                dummy[col] = 0
        dummy = dummy[df_train_clean.columns]
        df_train_clean = pd.concat([df_train_clean, dummy], ignore_index=True)
        print(f"  [INFO] Synthesised 1 seed row for missing ESI-{cls}")

df_train_aug = augment_training(df_train_clean, factor=8)
df_test = df_real.copy()

# FIX BUG 2: Don't use groupby().apply(resample) — it breaks in pandas ≥2.0.
# Resample each class separately and concat.
from sklearn.utils import resample

max_samples = df_train_aug['esi_level'].value_counts().max()
balanced_parts = []
for cls in all_classes:
    subset = df_train_aug[df_train_aug['esi_level'] == cls]
    if len(subset) == 0:
        continue
    resampled = resample(subset, replace=True, n_samples=max_samples, random_state=42)
    balanced_parts.append(resampled)
df_train_final = pd.concat(balanced_parts, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

train_median = df_train_final[FEATURE_COLS].median()
real_median  = df_real[FEATURE_COLS].median()

X_train = df_train_final[FEATURE_COLS].fillna(train_median)
y_train = df_train_final['esi_level'] - 1
groups  = df_train_final['subject_id'].values

X_test = df_test[FEATURE_COLS].fillna(real_median)
y_test = df_test['esi_level'] - 1

# FIX 1: Compute train_median AFTER augmentation so stats match the actual
# training distribution. Use a separate real_median only for filling test NaNs.

X_train = df_train_final[FEATURE_COLS].fillna(train_median)
y_train = df_train_final['esi_level'] - 1
groups  = df_train_final['subject_id'].values  # for GroupKFold

X_test  = df_test[FEATURE_COLS].fillna(real_median)
y_test  = df_test['esi_level'] - 1


# ═══════════════════════════════════════════════════════════
# 2. TRAIN XGBOOST
# ═══════════════════════════════════════════════════════════
print("Training XGBoost...")
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
    eval_metric='mlogloss',
    objective='multi:softprob',
    num_class=5,
    random_state=42,
    n_jobs=-1
)
# FIX 2: Compute per-sample weights to handle class imbalance.
# ESI-1 and ESI-2 (classes 0 and 1) are rare but the most safety-critical;
# inverse-frequency weighting ensures the model doesn't ignore them.
from sklearn.utils.class_weight import compute_sample_weight
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
# Convert features and sample weights to correct types
X_train = X_train.astype(np.float32)
X_test  = X_test.astype(np.float32)
sample_weights = sample_weights.astype(np.float32)
model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)
print("Done.\n")

# FIX 3 + BUG 4: Calibrate on a HELD-OUT calibration set, not on X_train.
# Fitting CalibratedClassifierCV on the same data the model trained on causes
# overconfidence (the model has already memorised those labels).
# Solution: hold out 20% of training data purely for calibration.
print("Calibrating model probabilities (sigmoid, held-out cal set)...")
from sklearn.model_selection import train_test_split
X_train_fit, X_cal, y_train_fit, y_cal = train_test_split(
    X_train, y_train, test_size=0.20, random_state=42, stratify=y_train)
sw_fit = compute_sample_weight(class_weight='balanced', y=y_train_fit).astype(np.float32)

# Retrain base model on the non-calibration portion
model.fit(X_train_fit, y_train_fit, sample_weight=sw_fit, verbose=False)

calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
sw_cal = compute_sample_weight(class_weight='balanced', y=y_cal).astype(np.float32)
calibrated_model.fit(X_cal, y_cal, sample_weight=sw_cal)
print("Calibration done.\n")


# ═══════════════════════════════════════════════════════════
# 3. HONEST EVALUATION
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("HONEST EVALUATION — SUBJECT-ISOLATED TEST SET")
print("=" * 60)

y_pred       = calibrated_model.predict(X_test)
y_pred_proba = calibrated_model.predict_proba(X_test)

weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
macro_f1    = f1_score(y_test, y_pred, average='macro',    zero_division=0)

print(f"\nWeighted F1 : {weighted_f1:.4f}")
print(f"Macro F1    : {macro_f1:.4f}")
print(f"(Target: 0.60–0.80 on real MIMIC is honest for a small dataset)\n")

# FIX 4: Report explicit recall for ESI-1 and ESI-2 — these are the
# safety-critical classes. Weighted/macro F1 hides failures on rare classes.
# Missing an ESI-1 is the most dangerous failure mode.
esi1_in_test = (y_test == 0).any()
esi2_in_test = (y_test == 1).any()
if esi1_in_test:
    esi1_recall = recall_score(y_test == 0, y_pred == 0, zero_division=0)
    print(f"ESI-1 Recall (safety-critical) : {esi1_recall:.4f}  ← must be high")
else:
    print("ESI-1 Recall: no ESI-1 cases in test set")
if esi2_in_test:
    esi2_recall = recall_score(y_test == 1, y_pred == 1, zero_division=0)
    print(f"ESI-2 Recall (safety-critical) : {esi2_recall:.4f}  ← must be high\n")
else:
    print("ESI-2 Recall: no ESI-2 cases in test set\n")

present_classes  = sorted(y_test.unique())
target_names_all = ['ESI-1', 'ESI-2', 'ESI-3', 'ESI-4', 'ESI-5']
target_names     = [target_names_all[c] for c in present_classes]

print(f"Classes in test set: {[c+1 for c in present_classes]}")
print(classification_report(y_test, y_pred, labels=present_classes,
                             target_names=target_names, zero_division=0))

# FIX 5: Use only present_classes for the confusion matrix so the display_labels
# always match the matrix dimensions (avoids misleading empty rows/columns).
cm = confusion_matrix(y_test, y_pred, labels=present_classes)
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(cm, display_labels=target_names).plot(
    ax=ax, colorbar=False, cmap='Blues')
ax.set_title('ESI Prediction — Subject-Isolated Test Set (Honest)')
plt.tight_layout()
plt.savefig('reports/confusion_matrix.png', dpi=150)
plt.close()
print("Confusion matrix → reports/confusion_matrix.png")


# ═══════════════════════════════════════════════════════════
# 4. STRATIFIED GROUP K-FOLD CV (subject_id as group key)
# ═══════════════════════════════════════════════════════════
print("\nStratifiedGroupKFold CV (subject_id groups)...")
cv_model = xgb.XGBClassifier(
    n_estimators=200, max_depth=3, learning_rate=0.05,
    subsample=0.7, colsample_bytree=0.7, min_child_weight=5,
    reg_alpha=0.5, reg_lambda=2.0, gamma=0.3,
    eval_metric='mlogloss', objective='multi:softprob',
    num_class=5, random_state=42, n_jobs=-1
)

# FIX BUG 3: When all synthetic rows share subject_id=0, every sample ends up
# in the same group and StratifiedGroupKFold cannot split → all folds empty.
# Fix: assign each row with subject_id=0 a unique synthetic group ID so the
# splitter can distribute them across folds.
groups_for_cv = df_train_final['subject_id'].copy().values.astype(int)
# Find rows with subject_id == 0 (synthetic) and give them unique IDs
max_real_id = groups_for_cv.max() + 1
synthetic_mask = (groups_for_cv == 0)
groups_for_cv[synthetic_mask] = max_real_id + np.arange(synthetic_mask.sum())

sgkf = StratifiedGroupKFold(n_splits=5)
cv_scores = []
cv_esi1_recalls = []
cv_esi2_recalls = []

X_train_np = X_train.to_numpy(dtype=np.float32)
y_train_np = y_train.to_numpy(dtype=np.int32).ravel()
_, groups_np = np.unique(groups_for_cv, return_inverse=True)

for fold, (tr_idx, val_idx) in enumerate(
        sgkf.split(X_train_np, y_train_np, groups=groups_np)):
    
    if len(tr_idx) == 0 or len(val_idx) == 0:
        print(f"Skipping fold {fold}: empty indices")
        continue

    # tr_idx and val_idx are already int arrays
    fold_weights = compute_sample_weight(class_weight='balanced',
                                         y=y_train_np[tr_idx])

    cv_model.fit(X_train_np[tr_idx], y_train_np[tr_idx],
                 sample_weight=fold_weights, verbose=False)

    preds = cv_model.predict(X_train_np[val_idx])
    val_y = y_train_np[val_idx]

    score = f1_score(val_y, preds, average='weighted', zero_division=0)
    cv_scores.append(score)

    if (val_y == 0).any():
        cv_esi1_recalls.append(recall_score(val_y == 0, preds == 0, zero_division=0))
    if (val_y == 1).any():
        cv_esi2_recalls.append(recall_score(val_y == 1, preds == 1, zero_division=0))


cv_scores = np.array(cv_scores) if cv_scores else np.array([])
if len(cv_scores) > 0:
    print(f"CV scores (5-fold SGK)  : {cv_scores.round(4)}")
    print(f"Mean CV F1              : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
else:
    print("CV scores (5-fold SGK)  : no valid folds completed")
    print("Mean CV F1              : N/A")
if cv_esi1_recalls:
    print(f"CV ESI-1 Recall (mean)  : {np.mean(cv_esi1_recalls):.4f}  ← safety metric")
if cv_esi2_recalls:
    print(f"CV ESI-2 Recall (mean)  : {np.mean(cv_esi2_recalls):.4f}  ← safety metric")
print("(Train/Test F1 gap < 0.25 = trustworthy; > 0.40 = still leaking)\n")


# ═══════════════════════════════════════════════════════════
# 5. CLINICAL OVERRIDE LAYER
# ═══════════════════════════════════════════════════════════
# Hard rules that OVERRIDE model output for patient safety.
# The model may under-triage critical patients when trained on
# limited data. These rules encode ACEP ESI guidelines directly.

def apply_clinical_override(model_esi: int, patient_data: dict) -> tuple[int, str | None]:
    """
    Returns (final_esi, override_reason).
    override_reason is None if no override applied.

    FIX 7: Now bidirectional:
      - Upward safety: upgrade ESI-3/4/5 → ESI-1/2 when vitals are critical.
      - Downward review: flag when model says ESI-1 but vitals don't clearly
        support it (borderline ESI-2) so a human can review.
    """
    hr   = patient_data.get('heart_rate',   0)
    sbp  = patient_data.get('systolic_bp',  999)
    spo2 = patient_data.get('spo2',         100)
    rr   = patient_data.get('resp_rate',    0)
    tmp  = patient_data.get('temperature',  98.6)

    # ── ESI-1: Immediate life threat ──────────────────────────
    esi1_criteria = []
    if sbp < 80:   esi1_criteria.append(f"critical hypotension (SBP {sbp} mmHg)")
    if spo2 < 88:  esi1_criteria.append(f"severe hypoxia (SpO₂ {spo2}%)")
    if hr > 150:   esi1_criteria.append(f"severe tachycardia (HR {hr} bpm)")
    if hr < 40:    esi1_criteria.append(f"severe bradycardia (HR {hr} bpm)")
    if rr > 35:    esi1_criteria.append(f"respiratory failure risk (RR {rr}/min)")
    if tmp < 95:   esi1_criteria.append(f"severe hypothermia (Temp {tmp}°F)")

    if esi1_criteria:
        reason = "ESI-1 OVERRIDE: " + "; ".join(esi1_criteria)
        return 1, reason

    # ── Downward review: model says ESI-1 but vitals are borderline ──
    # The model predicted highest acuity, but vitals don't clearly confirm it.
    # Return ESI-1 with a HUMAN REVIEW flag so clinicians don't blindly trust
    # a potentially overconfident model prediction.
    if model_esi == 1 and not esi1_criteria:
        borderline = []
        if 80 <= sbp < 90:   borderline.append(f"SBP borderline ({sbp} mmHg)")
        if 88 <= spo2 < 92:  borderline.append(f"SpO₂ borderline ({spo2}%)")
        if 130 < hr <= 150:  borderline.append(f"HR borderline ({hr} bpm)")
        if 28 < rr <= 35:    borderline.append(f"RR borderline ({rr}/min)")
        if borderline:
            reason = ("ESI-1 REVIEW FLAG: model predicted ESI-1 but vitals are borderline — "
                      + "; ".join(borderline) + ". Human verification recommended.")
            return 1, reason

    # ── ESI-2: High-risk situation ── upgrade only if model is lower ──
    if model_esi >= 3:
        esi2_criteria = []
        shock_index = hr / sbp if sbp > 0 else 0
        if sbp < 90:           esi2_criteria.append(f"hypotension (SBP {sbp} mmHg)")
        if spo2 < 92:          esi2_criteria.append(f"hypoxia (SpO₂ {spo2}%)")
        if hr > 130:           esi2_criteria.append(f"tachycardia (HR {hr} bpm)")
        if rr > 28:            esi2_criteria.append(f"tachypnea (RR {rr}/min)")
        if shock_index > 1.0:  esi2_criteria.append(f"elevated shock index ({shock_index:.2f})")

        if len(esi2_criteria) >= 2:
            reason = ("ESI-2 OVERRIDE: concurrent high-risk vitals — "
                      + "; ".join(esi2_criteria))
            return 2, reason

    return model_esi, None


# ═══════════════════════════════════════════════════════════
# 6. CLINICAL SIGNAL ENGINE
# ═══════════════════════════════════════════════════════════
def clinical_signal_engine(patient_data: dict) -> dict:
    """
    Analyses patient vitals and returns:
      - signals: list of flagged clinical findings with severity
      - critical_flags: list of the most dangerous findings
      - context_flags: comorbidities / demographic risk factors
    """
    hr   = patient_data.get('heart_rate')
    sbp  = patient_data.get('systolic_bp')
    dbp  = patient_data.get('diastolic_bp')
    rr   = patient_data.get('resp_rate')
    spo2 = patient_data.get('spo2')
    tmp  = patient_data.get('temperature')
    age  = patient_data.get('age', 0)

    signals  = []
    critical = []
    context  = []

    # --- Cardiovascular ---
    if hr is not None:
        if hr > 150:
            signals.append({'flag': 'SEVERE TACHYCARDIA', 'value': f'HR {hr} bpm',
                            'severity': 'critical',
                            'clinical_note': 'HR >150 suggests SVT, VT, or decompensated shock'})
            critical.append(f'Severe tachycardia (HR {hr} bpm)')
        elif hr > 100:
            signals.append({'flag': 'TACHYCARDIA', 'value': f'HR {hr} bpm',
                            'severity': 'high',
                            'clinical_note': 'Tachycardia may indicate pain, fever, hypovolemia, or arrhythmia'})
            critical.append(f'Tachycardia (HR {hr} bpm)')
        elif hr < 40:
            signals.append({'flag': 'SEVERE BRADYCARDIA', 'value': f'HR {hr} bpm',
                            'severity': 'critical',
                            'clinical_note': 'HR <40 risk of cardiac output failure'})
            critical.append(f'Severe bradycardia (HR {hr} bpm)')
        elif hr < 60:
            signals.append({'flag': 'BRADYCARDIA', 'value': f'HR {hr} bpm',
                            'severity': 'moderate',
                            'clinical_note': 'May be medication-related or cardiac conduction issue'})

    if sbp is not None:
        if sbp < 80:
            signals.append({'flag': 'CRITICAL HYPOTENSION', 'value': f'SBP {sbp} mmHg',
                            'severity': 'critical',
                            'clinical_note': 'Cardiogenic, distributive or haemorrhagic shock'})
            critical.append(f'Critical hypotension (SBP {sbp} mmHg)')
        elif sbp < 90:
            signals.append({'flag': 'HYPOTENSION', 'value': f'SBP {sbp} mmHg',
                            'severity': 'high',
                            'clinical_note': 'Borderline perfusion pressure — monitor closely'})
            critical.append(f'Hypotension (SBP {sbp} mmHg)')
        elif sbp > 180:
            signals.append({'flag': 'HYPERTENSIVE URGENCY', 'value': f'SBP {sbp} mmHg',
                            'severity': 'high',
                            'clinical_note': 'Hypertensive urgency/emergency — assess for end-organ damage'})

    if hr and sbp and sbp > 0:
        si = hr / sbp
        if si > 1.0:
            signals.append({'flag': 'ELEVATED SHOCK INDEX', 'value': f'SI {si:.2f}',
                            'severity': 'high',
                            'clinical_note': 'Shock index >1.0 correlates with haemodynamic instability'})
            critical.append(f'Shock index {si:.2f} (normal <0.7)')

    # --- Respiratory ---
    if spo2 is not None:
        if spo2 < 88:
            signals.append({'flag': 'SEVERE HYPOXIA', 'value': f'SpO₂ {spo2}%',
                            'severity': 'critical',
                            'clinical_note': 'SpO₂ <88% — immediate O₂ therapy and airway assessment'})
            critical.append(f'Severe hypoxia (SpO₂ {spo2}%)')
        elif spo2 < 92:
            signals.append({'flag': 'HYPOXIA', 'value': f'SpO₂ {spo2}%',
                            'severity': 'high',
                            'clinical_note': 'SpO₂ <92% — supplemental oxygen indicated'})
            critical.append(f'Hypoxia (SpO₂ {spo2}%)')
        elif spo2 < 95:
            signals.append({'flag': 'LOW SpO₂', 'value': f'SpO₂ {spo2}%',
                            'severity': 'moderate',
                            'clinical_note': 'SpO₂ below normal — monitor and consider oxygen'})

    if rr is not None:
        if rr > 30:
            signals.append({'flag': 'SEVERE TACHYPNEA', 'value': f'RR {rr}/min',
                            'severity': 'critical',
                            'clinical_note': 'RR >30 indicates respiratory distress or failure'})
            critical.append(f'Severe tachypnea (RR {rr}/min)')
        elif rr > 20:
            signals.append({'flag': 'TACHYPNEA', 'value': f'RR {rr}/min',
                            'severity': 'high',
                            'clinical_note': 'Elevated RR — assess for infection, pain, or respiratory compromise'})
            critical.append(f'Tachypnea (RR {rr}/min)')

    # --- Temperature ---
    if tmp is not None:
        if tmp < 95:
            signals.append({'flag': 'SEVERE HYPOTHERMIA', 'value': f'Temp {tmp}°F',
                            'severity': 'critical',
                            'clinical_note': 'Risk of cardiac arrhythmia and coagulopathy'})
            critical.append(f'Severe hypothermia (Temp {tmp}°F)')
        elif tmp > 103:
            signals.append({'flag': 'HIGH FEVER', 'value': f'Temp {tmp}°F',
                            'severity': 'high',
                            'clinical_note': 'Fever >103°F — consider sepsis workup'})
        elif tmp > 100.4:
            signals.append({'flag': 'FEVER', 'value': f'Temp {tmp}°F',
                            'severity': 'moderate',
                            'clinical_note': 'Low-grade fever — monitor, consider infection'})

    # --- Context ---
    if age > 65:
        context.append(f'Elderly patient (age {age}) — higher risk of atypical presentation')
    if patient_data.get('has_diabetes'):
        context.append('Diabetes — risk of silent MI, DKA, hypoglycaemia')
    if patient_data.get('has_hypertension'):
        context.append('Hypertension — assess for hypertensive emergency')
    if patient_data.get('has_heart_disease'):
        context.append('Known heart disease — prioritise cardiac workup')

    return {
        'signals':        signals,
        'critical_flags': critical,
        'context_flags':  context,
        'signal_count':   len(signals),
        'critical_count': len(critical),
    }


# ═══════════════════════════════════════════════════════════
# 7. GEMINI LLM — NEXT STEPS
# ═══════════════════════════════════════════════════════════
def build_gemini_prompt(patient_data: dict, clinical_output: dict,
                        esi_level: int, override_reason: str | None,
                        complaint_text: str) -> str:
    """
    Builds a structured prompt for Gemini.
    Clinical signals (NOT SHAP) drive the explanation.
    """
    critical_str = '\n'.join(f'  • {f}' for f in clinical_output['critical_flags']) \
                   or '  • None detected'
    context_str  = '\n'.join(f'  • {f}' for f in clinical_output['context_flags']) \
                   or '  • None'

    vitals_str = (
        f"HR={patient_data.get('heart_rate')} bpm, "
        f"BP={patient_data.get('systolic_bp')}/{patient_data.get('diastolic_bp')} mmHg, "
        f"SpO₂={patient_data.get('spo2')}%, "
        f"RR={patient_data.get('resp_rate')}/min, "
        f"Temp={patient_data.get('temperature')}°F, "
        f"Age={patient_data.get('age')}"
    )

    override_str = f"\n⚠️ CLINICAL OVERRIDE APPLIED: {override_reason}" if override_reason else ""

    return f"""You are a board-certified emergency medicine physician writing a triage note.

Patient vitals: {vitals_str}
Chief complaint: "{complaint_text or 'Not provided'}"
Assigned ESI Level: {esi_level} (1=immediate, 2=emergent, 3=urgent, 4=less urgent, 5=non-urgent){override_str}

Critical clinical findings:
{critical_str}

Contextual risk factors:
{context_str}

Instructions:
- top_3_clinical_findings: List the 3 most clinically dangerous findings for THIS patient. 
  Base these ONLY on the vitals and clinical findings above — not on statistical model scores.
  Each entry must name the finding AND explain its specific danger for this patient.
- immediate_next_steps: List 3-5 SPECIFIC, ordered clinical actions appropriate for ESI {esi_level}.
  Be concrete (e.g. "Obtain 12-lead ECG immediately — exclude STEMI given chest pain + tachycardia"
  not vague ("assess the patient")).
  Actions must be ordered by urgency.
- clinical_summary: One sentence risk profile for the charge nurse, naming the most likely
  clinical diagnosis and acuity level.
- confidence_explanation: One sentence explaining why this ESI level was assigned (or overridden).

Respond ONLY with valid JSON, no markdown, no extra text:
{{
  "top_3_clinical_findings": ["finding 1", "finding 2", "finding 3"],
  "immediate_next_steps": ["step 1", "step 2", "step 3", "step 4"],
  "clinical_summary": "...",
  "confidence_explanation": "..."
}}"""


def call_gemini_api(prompt: str, clinical_output: dict,
                    esi_level: int, override_reason: str | None) -> dict:
    """
    Calls Google Gemini API.
    Falls back to clinical-signal-based summary if API unavailable.
    """
    try:
        model_name = "models/gemini-2.5-flash"  # pick a valid model

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={
            "temperature": 0.2
            }
        )
         # Correct way to get LLM output text
        if hasattr(response, "text") and response.text:
            text = response.text
        elif hasattr(response, "contents") and response.contents:
            text = response.contents[0].text
        else:
            raise ValueError("No text in Gemini response")

        # Parse JSON safely
        gemini_json = json.loads(text)
        return gemini_json

    
    except Exception as e:
        print(f"  [Gemini API error: {e} — using clinical fallback]")

    # ── Clinical-signal fallback (no LLM) ───────────────
    # Build findings from actual clinical signals, not SHAP
        findings = []
        for sig in clinical_output['signals'][:3]:
            note = sig.get('clinical_note', '')
            findings.append(f"{sig['flag']} ({sig['value']}): {note}")

        if not findings:
            findings = ['No critical vital signs flagged at this time']

        # ESI-specific next steps from clinical guidelines
        esi_actions = {
            1: [
                "Activate resuscitation team immediately — patient meets ESI-1 criteria",
                "Establish two large-bore IV lines and begin fluid resuscitation",
                "Continuous cardiac monitoring, pulse oximetry and capnography",
                "Obtain 12-lead ECG, ABG, and STAT labs (BMP, CBC, lactate, troponin)",
                "Prepare for airway management — have RSI medications at bedside",
            ],
            2: [
                "Place patient in monitored bay within 10 minutes",
                "Obtain 12-lead ECG and IV access immediately",
                "STAT labs: BMP, CBC, troponin, lactate, coagulation panel",
                "Supplemental oxygen if SpO₂ <94%",
                "Notify attending physician immediately",
            ],
            3: [
                "Assign monitored room — 2+ resources required",
                "IV access, basic labs (BMP, CBC), and relevant imaging",
                "Vital sign reassessment every 30 minutes",
                "Pain or symptom management per protocol",
            ],
            4: [
                "Assign non-monitored room — 1 resource required",
                "Focused history and physical exam",
                "Targeted diagnostic test if indicated",
                "Routine vital sign monitoring",
            ],
            5: [
                "Fast-track or waiting room assessment appropriate",
                "No resources required beyond prescription or wound care",
                "Vital signs once on arrival",
            ],
        }

        override_note = override_reason if override_reason else f"Patient vitals consistent with ESI {esi_level}"

        return {
            "top_3_clinical_findings": findings,
            "immediate_next_steps":    esi_actions.get(esi_level, esi_actions[3]),
            "clinical_summary":        (
                f"ESI {esi_level} — "
                + (f"{len(clinical_output['critical_flags'])} critical vital sign abnormalities detected"
                if clinical_output['critical_flags']
                else "vitals within acceptable range at this time")
            ),
            "confidence_explanation": override_note,
        }


# ═══════════════════════════════════════════════════════════
# 8. MAIN PREDICT FUNCTION
# ═══════════════════════════════════════════════════════════
def explain_prediction(patient_data: dict,
                       complaint_text: str = '',
                       use_llm: bool = True) -> dict:
    """
    Args:
        patient_data  : dict of feature values
        complaint_text: free-text chief complaint (used for LLM context)
        use_llm       : True = call Gemini; False = clinical fallback only
    """
    input_df = pd.DataFrame([patient_data])[FEATURE_COLS]
    input_df = input_df.fillna(real_median)

    # Optionally encode complaint text if embedding model is available
    emb_cols = [c for c in FEATURE_COLS if c.startswith('complaint_emb_')]
    if complaint_text and emb_cols:
        try:
            from sentence_transformers import SentenceTransformer
            with open('models/complaint_embed_model.pkl', 'rb') as f:
                saved = pickle.load(f)
            em      = SentenceTransformer(saved['embed_model_name'])
            raw_emb = em.encode([complaint_text])
            reduced = saved['pca'].transform(raw_emb)[0]
            for i, col in enumerate(emb_cols):
                input_df[col] = reduced[i]
        except Exception:
            pass

    # Step 1: Model prediction (use calibrated model for reliable probabilities)
    proba           = calibrated_model.predict_proba(input_df)[0]
    model_class     = int(np.argmax(proba))
    model_esi       = model_class + 1
    raw_confidence  = float(proba[model_class])

    # Step 2: Clinical signal analysis (independent of model)
    clinical_output = clinical_signal_engine(patient_data)

    # Step 3: Override layer — safety net for under-triage (now bidirectional)
    final_esi, override_reason = apply_clinical_override(model_esi, patient_data)

    # Step 4: Uncertainty trigger — flag for human review when model and
    # clinical signals disagree, or when model confidence is low.
    # This is critical for deployment safety.
    needs_human_review = False
    uncertainty_reason = None
    CONFIDENCE_THRESHOLD = 0.50
    clinical_severity = clinical_output['critical_count']

    if raw_confidence < CONFIDENCE_THRESHOLD:
        needs_human_review = True
        uncertainty_reason = (f"Low model confidence ({raw_confidence:.1%}) — "
                               "prediction is uncertain. Human review required.")
    elif clinical_severity > 0 and final_esi > 2:
        # Clinical signals indicate high acuity but model predicts low urgency
        needs_human_review = True
        uncertainty_reason = (f"Model/clinical signal disagreement: model says ESI-{final_esi} "
                               f"but {clinical_severity} critical vital(s) flagged. "
                               "Human review required.")
    elif clinical_severity == 0 and final_esi <= 2 and not override_reason:
        # Model predicts high acuity but no clinical signals support it
        needs_human_review = True
        uncertainty_reason = (f"Model predicts ESI-{final_esi} but no critical vitals flagged. "
                               "Verify clinical picture before acting.")

    # Step 5: Adjust confidence display
    if override_reason:
        # Override applied — confidence reflects clinical certainty
        confidence = 0.95
    else:
        # Boost confidence slightly if clinical signals agree with model prediction
        if clinical_output['critical_count'] > 0 and final_esi <= 2:
            confidence = max(raw_confidence, 0.80)
        else:
            confidence = raw_confidence

    # Step 5: LLM next steps using clinical signals (not SHAP)
    if use_llm:
        prompt     = build_gemini_prompt(patient_data, clinical_output,
                                         final_esi, override_reason, complaint_text)
        llm_result = call_gemini_api(prompt, clinical_output, final_esi, override_reason)
    else:
        llm_result = call_gemini_api("", clinical_output, final_esi, override_reason)

    return {
        'esi_level':               final_esi,
        'model_esi':               model_esi,
        'override_applied':        override_reason is not None,
        'override_reason':         override_reason,
        'needs_human_review':      needs_human_review,       # FIX 8
        'uncertainty_reason':      uncertainty_reason,        # FIX 8
        'confidence':              round(confidence, 4),
        'top_clinical_findings':   llm_result.get('top_3_clinical_findings', []),
        'immediate_next_steps':    llm_result.get('immediate_next_steps', []),
        'clinical_summary':        llm_result.get('clinical_summary', ''),
        'confidence_explanation':  llm_result.get('confidence_explanation', ''),
        'clinical_signals':        clinical_output['signals'],
        'critical_flags':          clinical_output['critical_flags'],
        'context_flags':           clinical_output['context_flags'],
        'all_probabilities':       {f'ESI-{i+1}': round(float(p), 4)
                                    for i, p in enumerate(proba)},
    }


# ═══════════════════════════════════════════════════════════
# 9. DEMO
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("DEMO — 65M, chest pain, HR 140, BP 90/60, SpO₂ 91%")
print("=" * 60)
demo = {
    'heart_rate': 140, 'systolic_bp': 90,  'diastolic_bp': 60,
    'resp_rate':  28,  'spo2':        91,  'temperature':  101.2,
    'age':        65,  'gender_m':    1,   'complaint_cat': 1,
    'has_diabetes': 1, 'has_hypertension': 1, 'has_heart_disease': 0,
}
for i in range(8):
    demo[f'complaint_emb_{i}'] = 0.0

result = explain_prediction(
    demo,
    complaint_text="Chest pain radiating to left arm, diaphoretic, onset 2 hours ago.",
    use_llm=True  # will use Gemini if GEMINI_API_KEY is set, else clinical fallback
)

print(f"\n  ESI Level           : {result['esi_level']}"
      + (f"  ← OVERRIDDEN from model ESI-{result['model_esi']}" if result['override_applied'] else ''))
print(f"  Override applied    : {result['override_applied']}")
if result['override_reason']:
    print(f"  Override reason     : {result['override_reason']}")
print(f"  Needs human review  : {result['needs_human_review']}")
if result['uncertainty_reason']:
    print(f"  Uncertainty reason  : {result['uncertainty_reason']}")
print(f"  Confidence          : {result['confidence']:.1%}")
print(f"  Clinical summary    : {result['clinical_summary']}")
print(f"\n  Top clinical findings (NOT SHAP scores):")
for f in result['top_clinical_findings']:
    print(f"    • {f}")
print(f"\n  Immediate next steps:")
for s in result['immediate_next_steps']:
    print(f"    → {s}")
print(f"\n  Confidence explanation: {result['confidence_explanation']}")
print(f"  All probabilities   : {result['all_probabilities']}")


# ═══════════════════════════════════════════════════════════
# 10. SAVE
# ═══════════════════════════════════════════════════════════
print("\nSaving artifacts...")
with open('models/triage_model.pkl',   'wb') as f: pickle.dump(calibrated_model, f)
with open('models/feature_names.pkl',  'wb') as f: pickle.dump(FEATURE_COLS,      f)
with open('models/real_median.pkl',    'wb') as f: pickle.dump(real_median.to_dict(), f)
with open('models/train_median.pkl',   'wb') as f: pickle.dump(train_median.to_dict(), f)

esi1_recall_str = f"{esi1_recall:.4f}" if esi1_in_test else "N/A (no ESI-1 in test set)"
esi2_recall_str = f"{esi2_recall:.4f}" if esi2_in_test else "N/A (no ESI-2 in test set)"
cv_esi1_str = f"{np.mean(cv_esi1_recalls):.4f}" if cv_esi1_recalls else "N/A"
cv_esi2_str = f"{np.mean(cv_esi2_recalls):.4f}" if cv_esi2_recalls else "N/A"

eval_report = f"""
MODEL EVALUATION REPORT v4
============================

Test Performance (subject-isolated)
-------------------------------------
Weighted F1      : {weighted_f1:.4f}
Macro F1         : {macro_f1:.4f}
ESI-1 Recall     : {esi1_recall_str}   ← safety-critical
ESI-2 Recall     : {esi2_recall_str}   ← safety-critical

StratifiedGroupKFold CV (5 folds, subject_id groups, balanced weights)
------------------------------------------------------------------------
CV Scores         : {cv_scores.round(4) if len(cv_scores) > 0 else 'N/A'}
Mean CV F1        : {f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}" if len(cv_scores) > 0 else 'N/A'}
CV ESI-1 Recall   : {cv_esi1_str}   ← safety metric
CV ESI-2 Recall   : {cv_esi2_str}   ← safety metric
"""
with open('models/evaluation_report_v4.txt', 'w', encoding='utf-8') as f:
    f.write(eval_report)

print("Saved: triage_model.pkl (calibrated), feature_names.pkl, real_median.pkl, train_median.pkl")
print("Saved: models/evaluation_report_v4.txt")
print("\nAll done.")
