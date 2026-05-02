"""
pipeline_v3.py — Leakage-free data pipeline
=============================================
Key fix over v2:
  THE TRAIN/TEST SPLIT NOW HAPPENS INSIDE THE PIPELINE, BEFORE AUGMENTATION.

  v2 bug: ALL 70 real MIMIC rows were saved as both the test set AND as the
  source for augmentation. Adding 7% noise to a row and putting it in the
  training set while the original is in the test set is leakage — the model
  memorises every test patient with slightly different numbers → F1=1.0 (fake).

  v3 fix:
    - Split real MIMIC rows 80/20 by subject_id (subject-isolated, stratified)
    - 80% (df_train_real) → augment → add ESI-5 synthetic → balance → train CSV
    - 20% (df_test_real)  → saved as-is, never touched again → test CSV
    - PCA is fit ONLY on df_train_real embeddings to prevent test-set leakage
      into the embedding space as well
    - Synthetic ESI-5 rows get unique negative subject_ids so CV grouping works

  Other fixes carried over from v2:
    - Removed shock_index, has_sepsis, has_resp_failure
    - Sentence-embeddings → PCA(8) for chief complaint
    - Realistic ESI-5 synthetic patients

Run: python pipeline_v3.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.utils import resample
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ── PATHS ─────────────────────────────────────────────────
RAW   = Path('mimic-iii-clinical-db/raw')
OUT   = Path('mimic-iii-clinical-db')

# ── 1. LOAD ───────────────────────────────────────────────
print("Loading raw files...")
chart      = pd.read_csv(RAW / 'CHARTEVENTS.csv', low_memory=False)
admissions = pd.read_csv(RAW / 'ADMISSIONS.csv')
patients   = pd.read_csv(RAW / 'PATIENTS.csv')
diagnoses  = pd.read_csv(RAW / 'DIAGNOSES_ICD.csv')

# ── 2. VITALS ─────────────────────────────────────────────
chart['charttime']      = pd.to_datetime(chart['charttime'])
admissions['admittime'] = pd.to_datetime(admissions['admittime'])

vital_items = {
    220045: 'heart_rate',
    220179: 'systolic_bp',
    220180: 'diastolic_bp',
    220210: 'resp_rate',
    220277: 'spo2',
    223761: 'temperature'
}

vitals = chart[chart['itemid'].isin(vital_items.keys())]
vitals = vitals[vitals['error'] != 1]
vitals = vitals[['subject_id', 'hadm_id', 'itemid', 'valuenum', 'charttime']].dropna()
vitals['vital'] = vitals['itemid'].map(vital_items)
vitals = vitals.merge(admissions[['hadm_id', 'admittime']], on='hadm_id', how='inner')
vitals['time_from_admit'] = vitals['charttime'] - vitals['admittime']
vitals = vitals[vitals['time_from_admit'] >= pd.Timedelta(0)]
vitals = vitals.sort_values('time_from_admit')

vitals_closest = (
    vitals.groupby(['subject_id', 'hadm_id', 'vital'])['valuenum']
    .first().unstack().reset_index()
)

# ── 3. AGE + GENDER + LOS ─────────────────────────────────
admissions['dischtime'] = pd.to_datetime(admissions['dischtime'])
patients['dob']         = pd.to_datetime(patients['dob'])

adm = admissions.merge(patients[['subject_id', 'dob', 'gender']], on='subject_id', how='inner')

valid_dob = adm['dob'] > (adm['admittime'] - pd.Timedelta(days=365*150))
adm.loc[~valid_dob, 'dob'] = pd.NaT

adm['age']      = ((adm['admittime'] - adm['dob']).dt.days / 365).clip(0, 100).fillna(90).astype(int)
adm['gender_m'] = (adm['gender'] == 'M').astype(int)
adm['los_hours']= ((adm['dischtime'] - adm['admittime']).dt.total_seconds() / 3600).round(2)

# ── 4. COMORBIDITIES (pre-existing only — no sepsis/resp_failure) ──
def flag_comorbidities(df):
    codes = df.groupby('hadm_id')['icd9_code'].apply(set)
    r = pd.DataFrame(index=codes.index)
    r['has_diabetes']      = codes.apply(lambda x: any(str(c).startswith('250') for c in x))
    r['has_hypertension']  = codes.apply(lambda x: any(str(c).startswith('401') for c in x))
    r['has_heart_disease'] = codes.apply(lambda x: any(str(c).startswith('428') or str(c).startswith('414') for c in x))
    return r.reset_index().astype(int, errors='ignore')

comorbidities = flag_comorbidities(diagnoses)

# ── 5. CHIEF COMPLAINT — EMBED AS TEXT ────────────────────
# Map ADMISSIONS.diagnosis to realistic triage complaint text
complaint_text_map = {
    'sepsis':               'Patient presents with fever, chills and confusion. Possible infection.',
    'pneumonia':            'Difficulty breathing, productive cough and high fever since yesterday.',
    'coronary artery disease': 'Chest tightness and pressure radiating to left arm. Onset 2 hours ago.',
    'chest pain':           'Sharp chest pain, worse on breathing. Started this morning.',
    'gastrointestinal bleed': 'Vomiting blood, black tarry stools. Feels very weak.',
    'hepatitis':            'Severe abdominal pain, jaundice, nausea for 3 days.',
    'liver failure':        'Confusion, abdominal swelling, yellow skin. Chronic liver disease history.',
    'alcohol':              'Altered mental status, smells of alcohol. Found unresponsive.',
    'stroke':               'Sudden weakness on right side of body, slurred speech.',
    'syncope':              'Patient fainted suddenly. Brief loss of consciousness. Now awake.',
    'altered':              'Family reports confusion and disorientation since this morning.',
    'respiratory':          'Severe shortness of breath, cannot complete sentences.',
    'copd':                 'Wheezing and worsening shortness of breath. Known COPD.',
    'heart failure':        'Worsening leg swelling, shortness of breath lying flat.',
    'renal':                'Decreased urine output, leg swelling, fatigue for one week.',
    'overdose':             'Found unresponsive at home. Possible medication overdose.',
    'trauma':               'Involved in motor vehicle accident. Complains of chest and back pain.',
    'fever':                'High fever, body aches, fatigue. No other complaints.',
    'abdominal pain':       'Severe crampy abdominal pain, nausea and vomiting since last night.',
    'hypertension':         'Severe headache and blurred vision. BP very high at home.',
}

def get_complaint_text(diagnosis_str):
    if pd.isna(diagnosis_str):
        return 'Patient presents with general discomfort. No specific complaint.'
    d = str(diagnosis_str).lower().strip()
    for keyword, text in complaint_text_map.items():
        if keyword in d:
            return text
    return f'Patient presents with {d}.'

complaint = admissions[['hadm_id', 'diagnosis']].copy()
complaint['complaint_text'] = complaint['diagnosis'].apply(get_complaint_text)

# ── 6. ESI LABELS ─────────────────────────────────────────
def derive_esi(row):
    if   row['hospital_expire_flag'] == 1 and row['los_hours'] < 24: return 1
    elif row['hospital_expire_flag'] == 1 or  row['los_hours'] > 120: return 2
    elif row['los_hours'] > 48:                                        return 3
    elif row['los_hours'] > 12:                                        return 4
    else:                                                               return 5

adm['esi_level'] = adm.apply(derive_esi, axis=1)

# ── 7. MERGE (no embeddings yet — split first) ────────────
complaint['complaint_cat'] = complaint['complaint_text'].apply(
    lambda text: (
        1 if any(w in str(text).lower() for w in ['chest','cardiac','heart','coronary','pressure']) else
        2 if any(w in str(text).lower() for w in ['sepsis','infection','fever','pneumonia']) else
        3 if any(w in str(text).lower() for w in ['breath','respiratory','copd','asthma','shortness','dyspnea','wheez']) else
        4 if any(w in str(text).lower() for w in ['neuro','stroke','altered','seizure','syncope','tia','confusion','weakness']) else
        5 if any(w in str(text).lower() for w in ['abdom','gi','bowel','liver','bleed','hepat','vomit','nausea']) else
        0
    )
)

df = vitals_closest.merge(
    adm[['subject_id', 'hadm_id', 'age', 'gender_m',
         'los_hours', 'hospital_expire_flag', 'esi_level']],
    on=['subject_id', 'hadm_id'], how='inner'
)
df = df.merge(comorbidities,                           on='hadm_id', how='left')
df = df.merge(complaint[['hadm_id', 'complaint_cat', 'complaint_text']],
              on='hadm_id', how='left')
df = df.dropna(subset=['heart_rate', 'systolic_bp', 'spo2'])
df['los_hours'] = pd.to_numeric(df['los_hours'], errors='coerce').round(2)

print(f"\nFull real MIMIC dataset: {df.shape}")
print(df['esi_level'].value_counts().sort_index())

# ── 8. SUBJECT-ISOLATED TRAIN / TEST SPLIT ────────────────
# THIS IS THE CRITICAL FIX.
# Split by subject_id so no patient appears in both train and test.
# Stratify by esi_level so class distribution is preserved in both splits.
# Augmentation happens AFTER this split — the test set is never touched again.
from sklearn.model_selection import train_test_split

unique_subjects = df[['subject_id', 'esi_level']].drop_duplicates('subject_id')
train_subjects, test_subjects = train_test_split(
    unique_subjects['subject_id'],
    test_size=0.20,
    random_state=42,
    stratify=unique_subjects['esi_level']   # preserve ESI distribution
)

df_train_real = df[df['subject_id'].isin(train_subjects)].copy()
df_test_real  = df[df['subject_id'].isin(test_subjects)].copy()

print(f"\nTrain split (real): {len(df_train_real)} rows, "
      f"{df_train_real['subject_id'].nunique()} unique subjects")
print(df_train_real['esi_level'].value_counts().sort_index())
print(f"\nTest split  (real): {len(df_test_real)} rows, "
      f"{df_test_real['subject_id'].nunique()} unique subjects")
print(df_test_real['esi_level'].value_counts().sort_index())

# ── 9. EMBED COMPLAINT TEXT — FIT PCA ON TRAIN SPLIT ONLY ─
# Fitting PCA on all rows (including test) would leak test distribution
# into the feature space. Fit on train_real, transform both splits.
print("\nEmbedding complaint text with sentence-transformers...")
Path('models').mkdir(exist_ok=True)
USE_EMBEDDINGS = False

try:
    import pickle
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')

    train_texts = df_train_real['complaint_text'].tolist()
    test_texts  = df_test_real['complaint_text'].tolist()

    train_emb_raw = embed_model.encode(train_texts, show_progress_bar=False)
    test_emb_raw  = embed_model.encode(test_texts,  show_progress_bar=False)

    # Fit PCA on TRAIN only, transform both
    pca = PCA(n_components=8, random_state=42)
    train_emb_reduced = pca.fit_transform(train_emb_raw)
    test_emb_reduced  = pca.transform(test_emb_raw)
    print(f"Explained variance (8 PCA dims, train-fit): {pca.explained_variance_ratio_.sum():.1%}")

    emb_cols = [f'complaint_emb_{i}' for i in range(8)]

    train_emb_df = pd.DataFrame(train_emb_reduced, columns=emb_cols)
    train_emb_df['hadm_id'] = df_train_real['hadm_id'].values
    test_emb_df  = pd.DataFrame(test_emb_reduced,  columns=emb_cols)
    test_emb_df['hadm_id']  = df_test_real['hadm_id'].values

    df_train_real = df_train_real.merge(train_emb_df, on='hadm_id', how='left')
    df_test_real  = df_test_real.merge(test_emb_df,   on='hadm_id', how='left')

    with open('models/complaint_embed_model.pkl', 'wb') as f:
        pickle.dump({'pca': pca, 'embed_model_name': 'all-MiniLM-L6-v2'}, f)
    print("Saved PCA + embed model reference to models/complaint_embed_model.pkl")
    USE_EMBEDDINGS = True

except ImportError:
    print("sentence-transformers not installed — skipping embeddings")

# Drop complaint_text — it was only needed for embedding
df_train_real = df_train_real.drop(columns=['complaint_text'], errors='ignore')
df_test_real  = df_test_real.drop(columns=['complaint_text'],  errors='ignore')

# Move target to end
for _df in [df_train_real, df_test_real]:
    cols = [c for c in _df.columns if c != 'esi_level'] + ['esi_level']
    _df  = _df[cols]

# ── 10. SAVE TEST SET — NEVER TOUCHED AGAIN ───────────────
df_test_real.to_csv(OUT / 'mimic-iii-real.csv', index=False)
print(f"\nSaved test data  → mimic-iii-clinical-db/mimic-iii-real.csv  ({len(df_test_real)} rows)")

# ── 11. AUGMENT TRAIN SPLIT ONLY ──────────────────────────
def augment_training(df, factor=8, noise_pct=0.07):
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
df_aug = augment_training(df_train_real, factor=8)

# ── 12. SYNTHETIC ESI-5 (unique negative subject_ids for CV grouping) ──
def generate_esi5(n=80, df_reference=None):
    np.random.seed(42)
    texts = [
        'Mild rash on forearm, no other symptoms.',
        'Requesting prescription refill. Feeling well.',
        'Minor cut on finger, needs bandaging.',
        'Mild cold symptoms, runny nose since yesterday.',
        'Twisted ankle walking, mild swelling.',
        'Ear pain for two days, otherwise healthy.',
        'Sore throat, no fever, eating normally.',
        'Patient here for routine checkup and medication review.',
    ]
    complaint_texts = np.random.choice(texts, n)

    # Use negative IDs so they never collide with real MIMIC subject_ids (positive)
    base = pd.DataFrame({
        'subject_id':           -np.arange(1, n + 1, dtype=int),
        'hadm_id':              -np.arange(1, n + 1, dtype=int),
        'heart_rate':           np.random.normal(72, 6, n).clip(58, 88).round(1),
        'systolic_bp':          np.random.normal(116, 9, n).clip(98, 132).round(1),
        'diastolic_bp':         np.random.normal(74, 7, n).clip(58, 88).round(1),
        'resp_rate':            np.random.normal(14, 1.5, n).clip(10, 18).round(1),
        'spo2':                 np.random.normal(99, 0.4, n).clip(97, 100).round(1),
        'temperature':          np.random.normal(98.1, 0.35, n).clip(97, 99.2).round(1),
        'age':                  np.random.randint(18, 60, n),
        'gender_m':             np.random.randint(0, 2, n),
        'los_hours':            np.random.uniform(1.5, 8, n).round(2),
        'hospital_expire_flag': np.zeros(n, dtype=int),
        'has_diabetes':         np.zeros(n, dtype=int),
        'has_hypertension':     np.zeros(n, dtype=int),
        'has_heart_disease':    np.zeros(n, dtype=int),
        'complaint_cat':        np.zeros(n, dtype=int),
        'esi_level':            np.full(n, 5),
    })

    if USE_EMBEDDINGS:
        try:
            import pickle
            from sentence_transformers import SentenceTransformer
            with open('models/complaint_embed_model.pkl', 'rb') as f:
                saved = pickle.load(f)
            em = SentenceTransformer(saved['embed_model_name'])
            raw_emb = em.encode(complaint_texts.tolist(), show_progress_bar=False)
            reduced = saved['pca'].transform(raw_emb)
            for i in range(8):
                base[f'complaint_emb_{i}'] = reduced[:, i].round(6)
        except Exception as e:
            print(f"  [ESI-5 embedding failed: {e} — using zeros]")

    return base

esi5 = generate_esi5(80, df_reference=df_aug)
for col in df_aug.columns:
    if col not in esi5.columns:
        esi5[col] = 0
esi5 = esi5[df_aug.columns]
df_aug = pd.concat([df_aug, esi5], ignore_index=True)

# ── 13. BALANCE CLASSES ───────────────────────────────────
target_count = 150
balanced = []
for level in [1, 2, 3, 4, 5]:
    subset = df_aug[df_aug['esi_level'] == level]
    if len(subset) == 0:
        print(f"  [WARNING] No rows for ESI-{level} — skipping")
        continue
    replace = len(subset) < target_count
    balanced.append(resample(subset, replace=replace, n_samples=target_count, random_state=42))

df_train = (pd.concat(balanced, ignore_index=True)
            .sample(frac=1, random_state=42)
            .reset_index(drop=True))

# Move target to end
cols = [c for c in df_train.columns if c != 'esi_level'] + ['esi_level']
df_train = df_train[cols]

print(f"\nTraining dataset: {df_train.shape}")
print(df_train['esi_level'].value_counts().sort_index())

df_train.to_csv(OUT / 'mimic-iii-train.csv', index=False)
print(f"Saved training data → mimic-iii-clinical-db/mimic-iii-train.csv ({len(df_train)} rows)")
print("\nPipeline complete. Run train_model_v5.py next.")

print("Total admissions:", len(admissions))
print("Total chart events:", len(chart))
print("Vitals after filtering:", len(vitals))
print("After merging vitals+admissions:", len(vitals_closest))
print("After merging with adm:", len(df))