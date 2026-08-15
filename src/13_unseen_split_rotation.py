# ==========================================================
# MULTI-SPLIT KNOWN/UNSEEN ROTATION TEST
# Instead of relying on ONE choice of "6 known / 5 unseen"
# attack classes (which gave ROC-AUC 0.643), this runs several
# ROUNDS, each with a DIFFERENT random group of 6 known / 5
# unseen classes, and averages the ROC-AUC across all rounds.
#
# This gives an honest, defensible estimate of the Autoencoder's
# TRUE average generalization ability to unseen attack types -
# removing the risk that 0.643 was just a lucky single split.
#
# The honest average could come out LOWER than 0.643 - that's
# fine and expected; it's the more truthful number.
#
# Fully separate from your main paper pipeline.
# ==========================================================

N_ROUNDS = 5          # how many different known/unseen splits to test
N_KNOWN_ATTACKS = 6   # how many of the 11 attack classes are "known" each round

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys

class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data); s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

_log_file = open("multi_split_rotation_log.txt", "w", encoding="utf-8")
sys.stdout = _Tee(sys.stdout, _log_file)
sys.stderr = _Tee(sys.stderr, _log_file)

print(f"Starting multi-split known/unseen rotation test ({N_ROUNDS} rounds)")

import glob
import numpy as np
import pandas as pd
import joblib

# TensorFlow imported BEFORE sklearn, matching the fix for the
# import-order crash found earlier in this project.
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, average_precision_score

try:
    # ==========================================================
    # STEP 1 — LOAD AND CLEAN ONCE (reused across all rounds)
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

    class_counts = data['Label'].value_counts()
    valid_classes = class_counts[class_counts >= 10].index
    data = data[data['Label'].isin(valid_classes)]

    all_labels = sorted(data['Label'].unique())
    all_attack_labels = [c for c in all_labels if c != "BENIGN"]
    print(f"\nAll {len(all_attack_labels)} attack classes available: {all_attack_labels}")
    print(f"Will run {N_ROUNDS} rounds, each with a different random {N_KNOWN_ATTACKS}-known / "
          f"{len(all_attack_labels) - N_KNOWN_ATTACKS}-unseen split.\n")

    # ==========================================================
    # STEP 2 — RUN N_ROUNDS, EACH WITH A DIFFERENT RANDOM SPLIT
    # ==========================================================
    round_results = []

    for round_idx in range(N_ROUNDS):
        print("\n" + "=" * 70)
        print(f"ROUND {round_idx + 1} of {N_ROUNDS}")
        print("=" * 70)

        rng = np.random.RandomState(100 + round_idx)  # different seed each round
        known_attacks = list(rng.choice(all_attack_labels, size=N_KNOWN_ATTACKS, replace=False))
        unseen_attacks = [c for c in all_attack_labels if c not in known_attacks]

        print(f"Known classes this round: {known_attacks}")
        print(f"Unseen classes this round: {unseen_attacks}")

        known_mask = (data['Label'] == "BENIGN") | (data['Label'].isin(known_attacks))
        known_data = data[known_mask].copy()
        unseen_data = data[~known_mask].copy()

        le = LabelEncoder()
        known_data['Label'] = le.fit_transform(known_data['Label'])
        benign_class = le.transform(["BENIGN"])[0]

        X_known = known_data.drop("Label", axis=1)
        y_known = known_data["Label"]
        X_unseen = unseen_data.drop("Label", axis=1)[X_known.columns]

        X_train, X_test, y_train, y_test = train_test_split(
            X_known, y_known, test_size=0.2, random_state=42, stratify=y_known
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_unseen_scaled = scaler.transform(X_unseen)

        input_dim = X_train_scaled.shape[1]
        X_ae_train, X_ae_val = train_test_split(X_train_scaled, test_size=0.2, random_state=42)

        input_layer = Input(shape=(input_dim,))
        encoded = Dense(16, activation='relu')(input_layer)
        encoded = Dense(8, activation='relu')(encoded)
        decoded = Dense(16, activation='relu')(encoded)
        decoded = Dense(input_dim, activation='linear')(decoded)
        autoencoder = Model(inputs=input_layer, outputs=decoded)
        autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        autoencoder.fit(X_ae_train, X_ae_train, epochs=10, batch_size=512, shuffle=True, verbose=0)

        val_recon = autoencoder.predict(X_ae_val, verbose=0)
        val_mse = np.mean(np.power(X_ae_val - val_recon, 2), axis=1)
        threshold = np.percentile(val_mse, 95)

        test_recon = autoencoder.predict(X_test_scaled, verbose=0)
        test_mse = np.mean(np.power(X_test_scaled - test_recon, 2), axis=1)
        benign_test_mask = (y_test.values == benign_class)
        benign_test_mse = test_mse[benign_test_mask]
        benign_fpr = (benign_test_mse > threshold).mean()

        unseen_recon = autoencoder.predict(X_unseen_scaled, verbose=0)
        unseen_mse = np.mean(np.power(X_unseen_scaled - unseen_recon, 2), axis=1)
        unseen_detection = (unseen_mse > threshold).mean()

        y_binary = np.concatenate([np.ones(len(unseen_mse)), np.zeros(len(benign_test_mse))])
        scores_binary = np.concatenate([unseen_mse, benign_test_mse])
        roc_auc = roc_auc_score(y_binary, scores_binary)
        pr_auc = average_precision_score(y_binary, scores_binary)

        print(f"Round {round_idx + 1} result: ROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}  "
              f"FPR={benign_fpr:.4f}  Detection={unseen_detection:.4f}")

        round_results.append({
            "round": round_idx + 1,
            "known_attacks": known_attacks,
            "unseen_attacks": unseen_attacks,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "fpr": benign_fpr,
            "detection_rate": unseen_detection,
        })

    # ==========================================================
    # STEP 3 — AVERAGE ACROSS ALL ROUNDS (the honest number)
    # ==========================================================
    print("\n\n" + "=" * 70)
    print("FINAL SUMMARY — ALL ROUNDS")
    print("=" * 70)
    print(f"{'Round':<8}{'ROC-AUC':<12}{'PR-AUC':<12}{'FPR':<10}{'Detection':<12}")
    print("-" * 54)
    for r in round_results:
        print(f"{r['round']:<8}{r['roc_auc']:<12.4f}{r['pr_auc']:<12.4f}{r['fpr']:<10.4f}{r['detection_rate']:<12.4f}")

    roc_aucs = [r["roc_auc"] for r in round_results]
    pr_aucs = [r["pr_auc"] for r in round_results]

    mean_roc = np.mean(roc_aucs)
    std_roc = np.std(roc_aucs)
    mean_pr = np.mean(pr_aucs)
    std_pr = np.std(pr_aucs)

    print("\n" + "=" * 70)
    print("HONEST AVERAGE (across all random known/unseen splits)")
    print("=" * 70)
    print(f"Mean ROC-AUC: {mean_roc:.4f}  (std: {std_roc:.4f})")
    print(f"Mean PR-AUC:  {mean_pr:.4f}  (std: {std_pr:.4f})")
    print(f"Range: {min(roc_aucs):.4f} to {max(roc_aucs):.4f}")
    print(f"\n(For reference, the earlier single-split result was ROC-AUC 0.643.")
    print(f"This average of {N_ROUNDS} independent splits is the more honest, defensible number.)")

except Exception as e:
    import traceback
    print("\n\nSCRIPT FAILED WITH AN ERROR:")
    print(str(e))
    traceback.print_exc()

finally:
    _log_file.close()
