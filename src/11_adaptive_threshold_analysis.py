# ==========================================================
# LEAVE-ONE-CLASS-OUT — ADAPTIVE THRESHOLD ANALYSIS
# Same setup as before (Bot fully excluded from training), but
# this version:
#   1. Saves the trained models, so future experiments on this
#      same setup don't need to retrain from scratch again.
#   2. Checks reconstruction error for each of the 11 KNOWN
#      classes too, not just benign - tells us whether Bot's
#      error pattern resembles benign specifically or attacks
#      generally.
#   3. Computes the BEST POSSIBLE threshold (using the ground
#      truth we have in this controlled experiment) via ROC
#      analysis, to see the realistic ceiling for what
#      threshold-tuning alone could achieve.
# ==========================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys

class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

_log_file = open("adaptive_threshold_log.txt", "w", encoding="utf-8")
sys.stdout = _Tee(sys.stdout, _log_file)
sys.stderr = _Tee(sys.stderr, _log_file)

print("Starting leave-one-out adaptive threshold analysis (excluded class: Bot)")

import glob
import numpy as np
import pandas as pd
import joblib

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
    roc_curve
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

EXCLUDED_CLASS_NAME = "Bot"

try:
    # ==========================================================
    # STEP 1-5: SAME AS THE ORIGINAL LEAVE-ONE-OUT SCRIPT
    # ==========================================================
    data_dir = os.environ.get("CICIDS2017_DIR", "data/TrafficLabelling")
    path = os.path.join(data_dir, "*.csv")
    csv_files = glob.glob(path)
    print("CSV files found:", len(csv_files))

    df_list = []
    for file in csv_files:
        df = pd.read_csv(file, encoding="latin1", low_memory=False)
        df = df.sample(frac=0.2, random_state=42)
        df_list.append(df)

    data = pd.concat(df_list, ignore_index=True)
    data.columns = data.columns.str.strip()
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)

    drop_cols = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp']
    for col in drop_cols:
        if col in data.columns:
            data.drop(col, axis=1, inplace=True)

    data = data.astype(np.float32, errors='ignore')
    print("After cleaning:", data.shape)

    excluded_rows = data[data['Label'] == EXCLUDED_CLASS_NAME].copy()
    remaining_data = data[data['Label'] != EXCLUDED_CLASS_NAME].copy()
    print(f"Excluded class '{EXCLUDED_CLASS_NAME}': {len(excluded_rows)} samples set aside entirely")

    le = LabelEncoder()
    remaining_data['Label'] = le.fit_transform(remaining_data['Label'])
    class_counts = remaining_data['Label'].value_counts()
    valid_classes = class_counts[class_counts >= 10].index
    remaining_data = remaining_data[remaining_data['Label'].isin(valid_classes)]
    remaining_data['Label'] = le.inverse_transform(remaining_data['Label'])
    le = LabelEncoder()
    remaining_data['Label'] = le.fit_transform(remaining_data['Label'])
    n_classes = len(le.classes_)
    print(f"Classes in this 11-class model: {list(le.classes_)}")

    X = remaining_data.drop("Label", axis=1)
    y = remaining_data["Label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, y_train = resample(
        X_train, y_train, replace=False,
        n_samples=int(0.2 * len(X_train)), random_state=42, stratify=y_train
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Applying SMOTE...")
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_final, y_train_final = smote.fit_resample(X_train_scaled, y_train)

    X_excluded = excluded_rows.drop("Label", axis=1)[X.columns]
    X_excluded_scaled = scaler.transform(X_excluded)
    print(f"Prepared {len(X_excluded_scaled)} held-out '{EXCLUDED_CLASS_NAME}' samples")

    X_base_train, X_meta_train, y_base_train, y_meta_train = train_test_split(
        X_train_final, y_train_final, test_size=0.3, random_state=42, stratify=y_train_final
    )

    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=50, max_depth=12, n_jobs=1, random_state=42)
    rf.fit(X_base_train, y_base_train)

    print("Training XGBoost...")
    xgb = XGBClassifier(n_estimators=50, max_depth=6, learning_rate=0.1,
                         tree_method="hist", eval_metric="mlogloss", n_jobs=1)
    xgb.fit(X_base_train, y_base_train)

    print("Training DNN...")
    y_base_cat = to_categorical(y_base_train, n_classes)
    class_weights = compute_class_weight(class_weight="balanced",
                                          classes=np.unique(y_base_train), y=y_base_train)
    class_weights = dict(enumerate(class_weights))
    input_dim = X_base_train.shape[1]

    dnn_model = Sequential()
    dnn_model.add(Input(shape=(input_dim,)))
    dnn_model.add(Dense(128, activation="relu"))
    dnn_model.add(Dropout(0.3))
    dnn_model.add(Dense(64, activation="relu"))
    dnn_model.add(Dropout(0.3))
    dnn_model.add(Dense(n_classes, activation="softmax"))
    dnn_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    dnn_model.fit(X_base_train, y_base_cat, epochs=10, batch_size=512,
                  class_weight=class_weights, verbose=1)

    print("Training Autoencoder...")
    benign_class = le.transform(["BENIGN"])[0]
    X_train_benign = X_base_train[y_base_train == benign_class]
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(16, activation='relu')(input_layer)
    encoded = Dense(8, activation='relu')(encoded)
    decoded = Dense(16, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='linear')(decoded)
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    X_ae_train, X_ae_val = train_test_split(X_train_benign, test_size=0.2, random_state=42)
    autoencoder.fit(X_ae_train, X_ae_train, epochs=10, batch_size=512, shuffle=True, verbose=1)

    val_recon = autoencoder.predict(X_ae_val, verbose=0)
    val_mse = np.mean(np.power(X_ae_val - val_recon, 2), axis=1)
    default_threshold = np.percentile(val_mse, 95)
    print(f"Default (95th percentile benign) threshold: {default_threshold:.4f}")

    # Save everything so future experiments on this setup don't need retraining
    joblib.dump(rf, "leaveout_rf.pkl")
    joblib.dump(xgb, "leaveout_xgb.pkl")
    joblib.dump(scaler, "leaveout_scaler.pkl")
    joblib.dump(le, "leaveout_label_encoder.pkl")
    dnn_model.save("leaveout_dnn.keras")
    autoencoder.save("leaveout_autoencoder.keras")
    joblib.dump(X_excluded_scaled, "leaveout_bot_samples_scaled.pkl")
    joblib.dump(X_test_scaled, "leaveout_X_test_scaled.pkl")
    joblib.dump(y_test.values, "leaveout_y_test.pkl")
    print("Saved all models and prepared data for future reuse.")

    # ==========================================================
    # STEP 6 — RECONSTRUCTION ERROR FOR EACH KNOWN CLASS
    # Tells us: does Bot's error look like benign specifically,
    # or like attacks in general? This matters for whether an
    # attack-aware threshold could realistically help.
    # ==========================================================
    print("\n" + "=" * 60)
    print("RECONSTRUCTION ERROR BY KNOWN CLASS (test set)")
    print("=" * 60)

    recon_test = autoencoder.predict(X_test_scaled, verbose=0)
    mse_test_all = np.mean(np.power(X_test_scaled - recon_test, 2), axis=1)

    y_test_arr = y_test.values
    print(f"{'Class':<20}{'Mean MSE':<14}{'Median MSE':<14}{'n':<8}")
    print("-" * 56)
    for class_idx, class_name in enumerate(le.classes_):
        mask = (y_test_arr == class_idx)
        if mask.sum() == 0:
            continue
        class_mse = mse_test_all[mask]
        print(f"{class_name:<20}{class_mse.mean():<14.4f}{np.median(class_mse):<14.4f}{mask.sum():<8}")

    recon_excluded = autoencoder.predict(X_excluded_scaled, verbose=0)
    mse_excluded = np.mean(np.power(X_excluded_scaled - recon_excluded, 2), axis=1)
    print(f"{'Bot (UNSEEN, held out)':<20}{mse_excluded.mean():<14.4f}{np.median(mse_excluded):<14.4f}{len(mse_excluded):<8}")

    # ==========================================================
    # STEP 7 — BEST POSSIBLE THRESHOLD (uses ground truth we
    # have in THIS experiment only, to find the realistic ceiling)
    # ==========================================================
    print("\n" + "=" * 60)
    print("BEST-POSSIBLE THRESHOLD ANALYSIS (Bot vs. true benign)")
    print("=" * 60)

    benign_mask_test = (y_test_arr == benign_class)
    benign_mse_test = mse_test_all[benign_mask_test]

    y_binary = np.concatenate([np.ones(len(mse_excluded)), np.zeros(benign_mask_test.sum())])
    scores_binary = np.concatenate([mse_excluded, benign_mse_test])

    roc_auc = roc_auc_score(y_binary, scores_binary)
    pr_auc = average_precision_score(y_binary, scores_binary)
    print(f"ROC-AUC: {roc_auc:.4f}   PR-AUC: {pr_auc:.4f}")

    fpr, tpr, thresholds = roc_curve(y_binary, scores_binary)
    # Youden's J statistic: the threshold that best balances
    # true positive rate against false positive rate
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    best_tpr = tpr[best_idx]
    best_fpr = fpr[best_idx]

    print(f"\nBest possible threshold (Youden's J, found using ground truth): {best_threshold:.4f}")
    print(f"At this threshold: Detection rate (TPR) = {best_tpr:.4f}   False Positive Rate = {best_fpr:.4f}")
    print(f"(Compare to default 95th-percentile threshold {default_threshold:.4f}, which gave ~5.1% detection / ~4.9% FPR)")

    print("\nInterpretation:")
    if best_tpr - best_fpr < 0.15:
        print("Even the BEST possible threshold cannot meaningfully separate unseen Bot")
        print("traffic from benign traffic using reconstruction error alone. This means")
        print("threshold-tuning is NOT the fix - the reconstruction-error signal itself")
        print("does not carry enough information to detect this attack type when unseen.")
    else:
        print("The best possible threshold achieves meaningfully better separation than")
        print("the default threshold, suggesting a smarter/adaptive threshold could help.")

except Exception as e:
    import traceback
    print("\n\nSCRIPT FAILED WITH AN ERROR:")
    print(str(e))
    traceback.print_exc()

finally:
    _log_file.close()
