"""
Model Training, SHAP Explainability & Validation
Outputs:
  models/triage_model.pkl        ← trained XGBoost model
  models/feature_names.pkl       ← feature list (used by API)
  models/shap_explainer.pkl      ← SHAP TreeExplainer
  models/evaluation_report.txt  ← F1 scores, classification report
"""

import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── 0. SETUP ──────────────────────────────────────────────
Path("models").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)

# ── 1. LOAD PROCESSED DATASET ─────────────────────────────
print("Loading dataset...")
df = pd.read_csv('mimic-iii-clinical-db/mimic-iii-processed_dataset.csv')
print(f"Dataset shape: {df.shape}")
print(f"ESI distribution:\n{df['esi_level'].value_counts().sort_index()}\n")

# ── 2. DEFINE FEATURES ────────────────────────────────────
# Drop non-feature columns: IDs and leakage columns
DROP_COLS = ['subject_id', 'hadm_id', 'los_hours', 'hospital_expire_flag']

FEATURE_COLS = [c for c in df.columns
                if c not in DROP_COLS + ['esi_level']]

TARGET_COL = 'esi_level'

print(f"Features used ({len(FEATURE_COLS)}):")
for f in FEATURE_COLS:
    print(f"  - {f}")

X = df[FEATURE_COLS].copy()
y = df[TARGET_COL].copy()

# Fill any remaining NaNs with column median
X = X.fillna(X.median())

y_encoded = y - 1

# ── 3. TRAIN / TEST SPLIT ─────────────────────────────────
# Stratified split to preserve class balance in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,        # 80% train, 20% test
    random_state=42,
    stratify=y_encoded    # ensures each class is proportionally represented
)

print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
print(f"Train ESI distribution:\n{(y_train + 1).value_counts().sort_index()}")
print(f"Test ESI distribution:\n{(y_test + 1).value_counts().sort_index()}\n")

# ── 4. TRAIN XGBOOST ──────────────────────────────────────
print("Training XGBoost model...")

model = xgb.XGBClassifier(
    n_estimators=300,          # number of trees
    max_depth=5,               # tree depth — controls overfitting
    learning_rate=0.05,        # smaller = more conservative, less overfit
    subsample=0.8,             # use 80% of rows per tree (reduces overfit)
    colsample_bytree=0.8,      # use 80% of features per tree
    use_label_encoder=False,
    eval_metric='mlogloss',    # multi-class log loss
    objective='multi:softprob',
    num_class=5,
    random_state=42,
    n_jobs=-1                 
)

# Early stopping on a validation set during training
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train,
    test_size=0.15,
    random_state=42,
    stratify=y_train
)

model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    verbose=False
)

print("Training complete.\n")

# ── 5. EVALUATION ─────────────────────────────────────────
print("=" * 50)
print("EVALUATION ON HELD-OUT TEST SET")
print("=" * 50)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Weighted F1 — accounts for class imbalance
weighted_f1 = f1_score(y_test, y_pred, average='weighted')
macro_f1    = f1_score(y_test, y_pred, average='macro')

print(f"\nWeighted F1 Score : {weighted_f1:.4f}")
print(f"Macro F1 Score    : {macro_f1:.4f}")

# Per-class report
target_names = ['ESI-1', 'ESI-2', 'ESI-3', 'ESI-4', 'ESI-5']
report = classification_report(y_test, y_pred, target_names=target_names)
print(f"\nClassification Report:\n{report}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{cm}\n")

# Save confusion matrix plot
fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title('ESI Prediction — Confusion Matrix')
plt.tight_layout()
plt.savefig('reports/confusion_matrix.png', dpi=150)
plt.close()
print("Confusion matrix saved to reports/confusion_matrix.png")

# ── 6. CROSS-VALIDATION ───────────────────────────────────
print("\nRunning 5-fold cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y_encoded,
                             cv=cv, scoring='f1_weighted', n_jobs=-1)
print(f"CV Weighted F1 scores : {cv_scores.round(4)}")
print(f"Mean CV F1            : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")

# ── 7. SAVE EVALUATION REPORT ─────────────────────────────
report_text = f"""
MODEL EVALUATION REPORT
=======================
Dataset      : mimic-iii-processed_dataset.csv
Model        : XGBoost (n_estimators=300, max_depth=5, lr=0.05)
Train size   : {X_train.shape[0]}
Test size    : {X_test.shape[0]}
Features     : {len(FEATURE_COLS)}

Test Set Performance
--------------------
Weighted F1  : {weighted_f1:.4f}
Macro F1     : {macro_f1:.4f}

Cross-Validation (5-fold)
--------------------------
CV Scores    : {cv_scores.round(4)}
Mean CV F1   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}

Classification Report
----------------------
{report}
"""

with open('models/evaluation_report.txt', 'w') as f:
    f.write(report_text)
print("Evaluation report saved to models/evaluation_report.txt")

# ── 8. SHAP EXPLAINABILITY ────────────────────────────────
print("\nBuilding SHAP explainer...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# SHAP summary plot — shows which features matter most overall
plt.figure()
shap.summary_plot(
    shap_values,
    X_test,
    class_names=target_names,
    show=False,
    plot_size=(10, 6)
)
plt.tight_layout()
plt.savefig('reports/shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("SHAP summary plot saved to reports/shap_summary.png")

# ── 9. SHAP EXPLANATION FUNCTION ──────────────────────────
def explain_prediction(patient_data: dict) -> dict:
    """
    Given a patient dict with feature values, returns:
    - esi_level (1-5)
    - confidence (0-1)
    - top_risk_factors: list of 3 plain-English explanations

    Example input:
    {
        'heart_rate': 140, 'systolic_bp': 90, 'diastolic_bp': 60,
        'resp_rate': 28, 'spo2': 91, 'temperature': 101.2,
        'age': 65, 'gender_m': 1, 'shock_index': 1.56,
        'has_diabetes': 1, 'has_hypertension': 1,
        'has_heart_disease': 0, 'has_sepsis': 0,
        'has_resp_failure': 0, 'complaint_cat': 1
    }
    """
    # Build input dataframe in correct feature order
    input_df = pd.DataFrame([patient_data])[FEATURE_COLS]
    input_df = input_df.fillna(X.median())

    # Predict
    proba = model.predict_proba(input_df)[0]
    predicted_class = int(np.argmax(proba))
    esi_level = predicted_class + 1       # convert back from 0-indexed to 1-5
    confidence = float(proba[predicted_class])

    # SHAP values for this patient
    shap_vals = explainer(input_df)
    # shap_vals is a list of arrays, one per class
    # Use the predicted class's SHAP values
    if hasattr(shap_vals, "values"):
        # shape: (samples, features, classes)
        patient_shap = shap_vals.values[0][:, predicted_class]
    else:
        patient_shap = shap_vals[predicted_class][0]

    # Map features to SHAP contributions
    feature_contributions = dict(zip(FEATURE_COLS, patient_shap))

    # Sort by absolute contribution, take top 3
    top_features = sorted(
        feature_contributions.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:3]

    # Convert to plain-English labels
    feature_labels = {
        'heart_rate':        'Heart rate',
        'systolic_bp':       'Systolic blood pressure',
        'diastolic_bp':      'Diastolic blood pressure',
        'resp_rate':         'Respiratory rate',
        'spo2':              'Oxygen saturation (SpO2)',
        'temperature':       'Body temperature',
        'age':               'Patient age',
        'gender_m':          'Patient gender',
        'shock_index':       'Shock index (HR/BP)',
        'has_diabetes':      'Diabetes history',
        'has_hypertension':  'Hypertension history',
        'has_heart_disease': 'Heart disease history',
        'has_sepsis':        'Sepsis history',
        'has_resp_failure':  'Respiratory failure history',
        'complaint_cat':     'Chief complaint type',
    }

    top_risk_factors = []
    for feat, shap_val in top_features:
        label = feature_labels.get(feat, feat)
        value = patient_data.get(feat, 'N/A')
        direction = "elevated" if shap_val > 0 else "normal/low"
        top_risk_factors.append(
            f"{label} is {direction} (value: {value}, impact: {shap_val:+.3f})"
        )

    return {
        'esi_level':        esi_level,
        'confidence':       round(confidence, 4),
        'top_risk_factors': top_risk_factors,
        'all_probabilities': {
            f'ESI-{i+1}': round(float(p), 4)
            for i, p in enumerate(proba)
        }
    }

# ── 10. TEST THE EXPLANATION FUNCTION ─────────────────────
print("\nTesting explain_prediction() with a demo patient...")
demo_patient = {
    'heart_rate':        140,
    'systolic_bp':       90,
    'diastolic_bp':      60,
    'resp_rate':         28,
    'spo2':              91,
    'temperature':       101.2,
    'age':               65,
    'gender_m':          1,
    'shock_index':       140 / 90,
    'has_diabetes':      1,
    'has_hypertension':  1,
    'has_heart_disease': 0,
    'has_sepsis':        0,
    'has_resp_failure':  0,
    'complaint_cat':     1,   # cardiac/chest
}

result = explain_prediction(demo_patient)
print(f"\nDemo Patient Result:")
print(f"  ESI Level     : {result['esi_level']}")
print(f"  Confidence    : {result['confidence']:.1%}")
print(f"  Top risk factors:")
for r in result['top_risk_factors']:
    print(f"    • {r}")
print(f"  All probabilities: {result['all_probabilities']}")

# ── 11. SAVE MODEL + EXPLAINER ────────────────────────────
print("\nSaving model artifacts...")

with open('models/triage_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('models/feature_names.pkl', 'wb') as f:
    pickle.dump(FEATURE_COLS, f)

with open('models/shap_explainer.pkl', 'wb') as f:
    pickle.dump(explainer, f)

print("Model, feature names, and SHAP explainer saved to models/ directory.")
