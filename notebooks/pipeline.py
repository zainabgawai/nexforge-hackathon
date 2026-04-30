import pandas as pd
import numpy as np
from sklearn.utils import resample

# Load files
chart = pd.read_csv('mimic-iii-clinical-db/raw/CHARTEVENTS.csv', low_memory=False)
admissions = pd.read_csv('mimic-iii-clinical-db/raw/ADMISSIONS.csv')
patients = pd.read_csv('mimic-iii-clinical-db/raw/PATIENTS.csv')
diagnoses  = pd.read_csv('mimic-iii-clinical-db/raw/DIAGNOSES_ICD.csv')

# Convert time columns to datetime
chart['charttime'] = pd.to_datetime(chart['charttime'])
admissions['admittime'] = pd.to_datetime(admissions['admittime'])

# Define the itemids for vitals
vital_items = {
    220045: 'heart_rate',
    220179: 'systolic_bp',
    220180: 'diastolic_bp',
    220210: 'resp_rate',
    220277: 'spo2',
    223761: 'temperature'
}

# Filter relevant rows
vitals = chart[chart['itemid'].isin(vital_items.keys())]
vitals = vitals[vitals['error'] != 1]

# Keep necessary columns (IMPORTANT: include charttime)
vitals = vitals[['subject_id', 'hadm_id', 'itemid', 'valuenum', 'charttime']].dropna()

# Map itemid to readable vital names
vitals['vital'] = vitals['itemid'].map(vital_items)

# Merge with admissions to get admittime
vitals = vitals.merge(
    admissions[['hadm_id', 'admittime']],
    on='hadm_id',
    how='inner'
)

# Compute time difference from admission
vitals['time_from_admit'] = vitals['charttime'] - vitals['admittime']

# Keep only measurements AFTER admission
vitals = vitals[vitals['time_from_admit'] >= pd.Timedelta(0)]

# Sort so closest to admission comes first
vitals = vitals.sort_values('time_from_admit')

# Take the closest (first) measurement per patient/admission/vital
vitals_closest = (
    vitals
    .groupby(['subject_id', 'hadm_id', 'vital'])['valuenum']
    .first()
    .unstack()
    .reset_index()
)

# ── 3. PATIENT AGE ───────────────────────────────────────
admissions['admittime'] = pd.to_datetime(admissions['admittime'])
admissions['dischtime'] = pd.to_datetime(admissions['dischtime'])
patients['dob'] = pd.to_datetime(patients['dob'])

adm = admissions.merge(
    patients[['subject_id', 'dob', 'gender']],
    on='subject_id',
    how='inner'
)

# Remove absurd DOBs (older than 150 years from admission)
valid_dob_mask = adm['dob'] > (adm['admittime'] - pd.Timedelta(days=365*150))

adm.loc[~valid_dob_mask, 'dob'] = pd.NaT

age_days = (adm['admittime'] - adm['dob']).dt.days
# Convert to years 
adm['age'] = (age_days / 365).clip(0, 100)

# Fill missing (invalid DOBs → treated as elderly)
adm['age'] = adm['age'].fillna(90).astype(int)

# Gender encoding
adm['gender_m'] = (adm['gender'] == 'M').astype(int)

# Length of stay
adm['los_hours'] = (
    (adm['dischtime'] - adm['admittime']).dt.total_seconds() / 3600
)

# ── 4. COMORBIDITIES ─────────────────────────────────────
def flag_comorbidities(df):
    hadm_codes = df.groupby('hadm_id')['icd9_code'].apply(set)
    result = pd.DataFrame(index=hadm_codes.index)
    result['has_diabetes']      = hadm_codes.apply(lambda x: any(str(c).startswith('250') for c in x))
    result['has_hypertension']  = hadm_codes.apply(lambda x: any(str(c).startswith('401') for c in x))
    result['has_heart_disease'] = hadm_codes.apply(lambda x: any(str(c).startswith('428') or str(c).startswith('414') for c in x))
    result['has_sepsis']        = hadm_codes.apply(lambda x: any(str(c).startswith('995') for c in x))
    result['has_resp_failure']  = hadm_codes.apply(lambda x: any(str(c).startswith('518') for c in x))
    return result.reset_index().astype(int, errors='ignore')

comorbidities = flag_comorbidities(diagnoses)

# ── 5. CHIEF COMPLAINT from ADMISSIONS ───────────────────
complaint = admissions[['hadm_id', 'diagnosis']].copy()
complaint['diagnosis'] = complaint['diagnosis'].fillna('unknown').str.lower()

# Keyword → numeric category
def encode_complaint(text):
    if any(w in text for w in ['chest', 'cardiac', 'heart', 'coronary']):
        return 1
    elif any(w in text for w in ['sepsis', 'infection', 'fever', 'pneumonia']):
        return 2
    elif any(w in text for w in ['breath', 'respiratory', 'copd', 'asthma', 'shortness', 'dyspnea']):  
        return 3
    elif any(w in text for w in ['neuro', 'stroke', 'altered', 'seizure', 'syncope', 'tia']):    
        return 4
    elif any(w in text for w in ['abdom', 'gi', 'bowel', 'liver', 'bleed', 'hepat']):      
        return 5
    else:
        return 0

complaint['complaint_cat'] = complaint['diagnosis'].apply(encode_complaint)

# ── 6. DERIVE ESI LABEL ──────────────────────────────────
def derive_esi(row):
    if row['hospital_expire_flag'] == 1 and row['los_hours'] < 24:
        return 1
    elif row['hospital_expire_flag'] == 1 or row['los_hours'] > 120:
        return 2
    elif row['los_hours'] > 48:
        return 3
    elif row['los_hours'] > 12:
        return 4
    else:
        return 5

adm['esi_level'] = adm.apply(derive_esi, axis=1)

# ── 7. MERGE EVERYTHING ──────────────────────────────────
df = vitals_closest.merge(adm[['subject_id', 'hadm_id', 'age', 'gender_m',
                               'los_hours', 'hospital_expire_flag', 'esi_level']],
                         on=['subject_id', 'hadm_id'], how='inner')
df = df.merge(comorbidities, on='hadm_id', how='left')
df = df.merge(complaint[['hadm_id', 'complaint_cat']], on='hadm_id', how='left')

# Engineered feature: shock index
df['shock_index'] = df['heart_rate'] / df['systolic_bp'].replace(0, np.nan)

# Drop rows with too many missing vitals
df = df.dropna(subset=['heart_rate', 'systolic_bp', 'spo2'])
df['los_hours'] = pd.to_numeric(df['los_hours'], errors='coerce').round(2)

# ── 8. DATA AUGMENTATION ──────────────────────────────────
def augment_data(df, factor=5):
    numeric_cols = [
        'heart_rate', 'systolic_bp', 'diastolic_bp',
        'resp_rate', 'spo2', 'temperature',
        'age', 'shock_index'
    ]

    augmented = []

    for _ in range(factor):
        noisy = df.copy()

        for col in numeric_cols:
            if col in noisy.columns:
                std = noisy[col].std() * 0.05  # 5% noise
                noisy[col] = (noisy[col] +
                               np.random.normal(0, std, len(noisy))).round(2)

        augmented.append(noisy)

    return pd.concat([df] + augmented, ignore_index=True)


df_augmented = augment_data(df, factor=5)

# Recompute shock index after augmentation
df_augmented['shock_index'] = (
    df_augmented['heart_rate'] /
    df_augmented['systolic_bp'].replace(0, np.nan)
)

# ── REORDER COLUMNS (move target to end) ──────────────────
cols = [c for c in df_augmented.columns if c != 'esi_level'] + ['esi_level']
df_augmented = df_augmented[cols]

print(f"\nAugmented dataset: {df_augmented.shape}")
print(df_augmented['esi_level'].value_counts().sort_index())

# ── 9. ADD SYNTHETIC ESI 5 PATIENTS ───────────────────────
def generate_esi5(n=60):
    np.random.seed(42)
    return pd.DataFrame({
        'subject_id':            np.zeros(n, dtype=int),
        'hadm_id':               np.zeros(n, dtype=int),
        'heart_rate':            np.random.normal(75, 8, n).clip(60, 95).round(1),
        'systolic_bp':           np.random.normal(118, 10, n).clip(100, 135).round(1),
        'diastolic_bp':          np.random.normal(75, 8, n).clip(60, 90).round(1),
        'resp_rate':             np.random.normal(14, 2, n).clip(10, 18).round(1),
        'spo2':                  np.random.normal(99, 0.5, n).clip(97, 100).round(1),
        'temperature':           np.random.normal(98.2, 0.4, n).clip(97, 99.5).round(1),
        'age':                   np.random.randint(18, 65, n),
        'gender_m':              np.random.randint(0, 2, n),
        'los_hours':             np.random.uniform(2, 10, n).round(2),
        'hospital_expire_flag':  np.zeros(n, dtype=int),
        'has_diabetes':          np.zeros(n, dtype=int),
        'has_hypertension':      np.zeros(n, dtype=int),
        'has_heart_disease':     np.zeros(n, dtype=int),
        'has_sepsis':            np.zeros(n, dtype=int),
        'has_resp_failure':      np.zeros(n, dtype=int),
        'complaint_cat':         np.random.choice([0, 1, 2, 3, 4, 5], n),
        'esi_level':             np.full(n, 5),
    })

esi5 = generate_esi5(60)
esi5['shock_index'] = esi5['heart_rate'] / esi5['systolic_bp']

# Match column order before concat
esi5 = esi5[df_augmented.columns]

df_augmented = pd.concat([df_augmented, esi5], ignore_index=True)

# ── 10. BALANCE CLASSES ────────────────────────────────────
target_count = 150  # target per class

balanced_dfs = []
for level in [1, 2, 3, 4, 5]:
    subset = df_augmented[df_augmented['esi_level'] == level]
    if len(subset) < target_count:
        subset = resample(subset, replace=True,
                          n_samples=target_count, random_state=42)
    else:
        subset = resample(subset, replace=False,
                          n_samples=target_count, random_state=42)
    balanced_dfs.append(subset)

df_final = (
    pd.concat(balanced_dfs, ignore_index=True)
    .sample(frac=1, random_state=42)
    .reset_index(drop=True)
)

print(f"\nFinal balanced dataset: {df_final.shape}")
print(df_final['esi_level'].value_counts().sort_index())

df_final.to_csv(
    'mimic-iii-clinical-db/mimic-iii-processed_dataset.csv',
    index=False
)

print("\nSaved to mimic-iii-clinical-db/mimic-iii-processed_dataset.csv")