import os
# ==========================================================
# PART 1 — DATA PREPROCESSING (NO-PCA VERSION)
# LOAD -> SAMPLE -> CLEAN -> REMOVE RARE CLASSES -> SPLIT
# -> REDUCE TRAIN SIZE -> SCALE -> SMOTE -> SAVE
# (PCA step removed, based on ablation results showing PCA
#  reduced performance for every model tested)
# ==========================================================

import pandas as pd
import numpy as np
import glob
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

print("Starting IDS DATA PREPROCESSING (no-PCA version)")

# ==========================================================
# 1 — LOAD DATASET (WITH SAMPLING)
# ==========================================================
data_dir = os.environ.get("CICIDS2017_DIR", "data/TrafficLabelling")
path = os.path.join(data_dir, "*.csv")
csv_files = glob.glob(path)
print("CSV files found:", len(csv_files))

df_list = []
for file in csv_files:
    print("Loading:", file)
    df = pd.read_csv(file, encoding="latin1", low_memory=False)
    df = df.sample(frac=0.2, random_state=42)
    df_list.append(df)

data = pd.concat(df_list, ignore_index=True)
print("Dataset after sampling:", data.shape)

# ==========================================================
# 2 — CLEAN DATA
# ==========================================================
data.columns = data.columns.str.strip()
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

drop_cols = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp']
for col in drop_cols:
    if col in data.columns:
        data.drop(col, axis=1, inplace=True)

print("After cleaning:", data.shape)
data = data.astype(np.float32, errors='ignore')

# ==========================================================
# 3 — LABEL ENCODING
# ==========================================================
le = LabelEncoder()
data['Label'] = le.fit_transform(data['Label'])

# ==========================================================
# 4 — REMOVE RARE CLASSES
# ==========================================================
class_counts = data['Label'].value_counts()
valid_classes = class_counts[class_counts >= 10].index
data = data[data['Label'].isin(valid_classes)]
print("\nAfter removing rare classes:", data.shape)

data['Label'] = le.inverse_transform(data['Label'])
le = LabelEncoder()
data['Label'] = le.fit_transform(data['Label'])

X = data.drop("Label", axis=1)
y = data["Label"]

# ==========================================================
# 5 — TRAIN TEST SPLIT
# ==========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# ==========================================================
# 6 — REDUCE TRAIN SIZE (MEMORY SAFE)
# ==========================================================
print("\nChecking per-class counts BEFORE reduction:")
print(y_train.value_counts())

X_train, y_train = resample(
    X_train, y_train,
    replace=False,
    n_samples=int(0.2 * len(X_train)),
    random_state=42,
    stratify=y_train
)

print("\nReduced Train shape:", X_train.shape)
print("Checking per-class counts AFTER reduction:")
print(pd.Series(y_train).value_counts())

# ==========================================================
# 7 — SCALING
# ==========================================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Scaling completed")

# ==========================================================
# 8 — APPLY SMOTE (kept — ablation showed SMOTE+class_weight
# together works best; only PCA is being removed here)
# ==========================================================
print("\nApplying SMOTE...")
smote = SMOTE(random_state=42, k_neighbors=3)
X_train, y_train = smote.fit_resample(X_train, y_train)
print("After SMOTE:", X_train.shape)

# ==========================================================
# 9 — PCA STEP REMOVED
# (This is the only structural change from your original
#  part1.py — ablation results showed PCA reduced macro-F1
#  for every model tested: RF 0.79->0.88, XGBoost 0.75->0.90,
#  DNN 0.70->0.79, all WITHOUT PCA scoring higher.)
# ==========================================================

# ==========================================================
# 10 — SAVE PROCESSED DATA
# (saved with "_nopca" suffix so these don't overwrite your
#  original PCA-based files or your class-weight files)
# ==========================================================
joblib.dump(X_train, "X_train_nopca.pkl")
joblib.dump(X_test, "X_test_nopca.pkl")
joblib.dump(y_train, "y_train_nopca.pkl")
joblib.dump(y_test, "y_test_nopca.pkl")
joblib.dump(scaler, "scaler_nopca.pkl")
joblib.dump(le, "label_encoder_nopca.pkl")

print("\nPreprocessed dataset (no-PCA version) saved successfully")
print("Final feature count:", X_train.shape[1])