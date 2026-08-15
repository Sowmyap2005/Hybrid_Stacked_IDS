# ==========================================================
# AUTOENCODER LOSS FUNCTION COMPARISON: MSE vs MAE
# Trains the SAME autoencoder architecture on the SAME benign
# training data twice - once with MSE loss, once with MAE loss -
# then applies the SAME 95th-percentile thresholding method to
# both, and compares attack detection rate and false positive
# rate on the full test set. The stronger loss function can then
# be selected for the final model.
#
# Uses the already-saved no-PCA data - no need to retrain
# RF/XGBoost/DNN, only the autoencoder is trained (twice).
# ==========================================================

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

_log_file = open("autoencoder_loss_comparison_log.txt", "w", encoding="utf-8")
sys.stdout = _Tee(sys.stdout, _log_file)
sys.stderr = _Tee(sys.stderr, _log_file)

print("Starting Autoencoder loss function comparison: MSE vs MAE")

import joblib
import numpy as np

# TensorFlow imported BEFORE sklearn, to avoid the import-order
# crash found earlier in this project.
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

try:
    # ==========================================================
    # LOAD DATA (same files your main pipeline uses)
    # ==========================================================
    X_train = joblib.load("X_train_nopca.pkl")
    y_train = joblib.load("y_train_nopca.pkl")
    X_test = joblib.load("X_test_nopca.pkl")
    y_test = joblib.load("y_test_nopca.pkl")
    le = joblib.load("label_encoder_nopca.pkl")

    benign_class = le.transform(["BENIGN"])[0]

    # Same 70/30 base/meta split used for the main pipeline, so
    # the autoencoder trains on the exact same rows as before
    X_base_train, X_meta_train, y_base_train, y_meta_train = train_test_split(
        X_train, y_train, test_size=0.3, random_state=42, stratify=y_train
    )

    X_train_benign = X_base_train[y_base_train == benign_class]
    input_dim = X_base_train.shape[1]

    # Held-out benign validation slice for threshold-setting
    # (same method as the main paper's threshold analysis)
    X_ae_train, X_ae_val = train_test_split(X_train_benign, test_size=0.2, random_state=42)

    # ==========================================================
    # HELPER: build, train, and evaluate one autoencoder with a
    # given loss function. Everything else stays identical.
    # ==========================================================
    def run_autoencoder(loss_name):
        print(f"\n{'='*60}")
        print(f"Training Autoencoder with loss = {loss_name}")
        print(f"{'='*60}")

        input_layer = Input(shape=(input_dim,))
        encoded = Dense(16, activation='relu')(input_layer)
        encoded = Dense(8, activation='relu')(encoded)
        decoded = Dense(16, activation='relu')(encoded)
        decoded = Dense(input_dim, activation='linear')(decoded)
        model = Model(inputs=input_layer, outputs=decoded)
        model.compile(optimizer=Adam(learning_rate=0.001), loss=loss_name)

        model.fit(X_ae_train, X_ae_train, epochs=10, batch_size=512,
                  shuffle=True, verbose=1)

        # --- Set threshold using the SAME method: 95th percentile
        # of error on the held-out benign validation slice ---
        val_recon = model.predict(X_ae_val, verbose=0)
        if loss_name == "mse":
            val_error = np.mean(np.power(X_ae_val - val_recon, 2), axis=1)
        else:  # mae
            val_error = np.mean(np.abs(X_ae_val - val_recon), axis=1)

        threshold = np.percentile(val_error, 95)
        print(f"Threshold (95th percentile, {loss_name}): {threshold:.4f}")

        # --- Apply to full test set ---
        test_recon = model.predict(X_test, verbose=0)
        if loss_name == "mse":
            test_error = np.mean(np.power(X_test - test_recon, 2), axis=1)
        else:
            test_error = np.mean(np.abs(X_test - test_recon), axis=1)

        y_test_binary = (y_test != benign_class).astype(int)  # 1 = attack, 0 = benign
        test_pred_anomaly = (test_error > threshold).astype(int)

        benign_mask = (y_test == benign_class)
        fpr = test_pred_anomaly[benign_mask].mean()
        attack_mask = ~benign_mask
        detection_rate = test_pred_anomaly[attack_mask].mean()

        roc_auc = roc_auc_score(y_test_binary, test_error)
        pr_auc = average_precision_score(y_test_binary, test_error)

        print(f"False Positive Rate on benign: {fpr:.4f}")
        print(f"Overall attack detection rate: {detection_rate:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}   PR-AUC: {pr_auc:.4f}")

        # Per-class detection rate too, for a fuller comparison
        print(f"\nDetection rate per attack class ({loss_name}):")
        per_class = {}
        for class_idx, class_name in enumerate(le.classes_):
            if class_idx == benign_class:
                continue
            class_mask = (y_test == class_idx)
            if class_mask.sum() == 0:
                continue
            rate = test_pred_anomaly[class_mask].mean()
            per_class[class_name] = rate
            print(f"  {class_name:<28} n={class_mask.sum():<7} detection_rate={rate:.4f}")

        return {
            "loss": loss_name, "threshold": threshold, "fpr": fpr,
            "detection_rate": detection_rate, "roc_auc": roc_auc,
            "pr_auc": pr_auc, "per_class": per_class
        }

    # ==========================================================
    # RUN BOTH VERSIONS
    # ==========================================================
    results_mse = run_autoencoder("mse")
    results_mae = run_autoencoder("mae")

    # ==========================================================
    # FINAL COMPARISON
    # ==========================================================
    print("\n" + "=" * 60)
    print("FINAL COMPARISON: MSE vs MAE")
    print("=" * 60)
    print(f"{'Metric':<30}{'MSE':<15}{'MAE':<15}")
    print("-" * 60)
    print(f"{'False Positive Rate':<30}{results_mse['fpr']:<15.4f}{results_mae['fpr']:<15.4f}")
    print(f"{'Attack Detection Rate':<30}{results_mse['detection_rate']:<15.4f}{results_mae['detection_rate']:<15.4f}")
    print(f"{'ROC-AUC':<30}{results_mse['roc_auc']:<15.4f}{results_mae['roc_auc']:<15.4f}")
    print(f"{'PR-AUC':<30}{results_mse['pr_auc']:<15.4f}{results_mae['pr_auc']:<15.4f}")

    print("\nPer-class detection rate comparison:")
    print(f"{'Class':<28}{'MSE':<12}{'MAE':<12}")
    print("-" * 52)
    for class_name in results_mse["per_class"]:
        mse_val = results_mse["per_class"][class_name]
        mae_val = results_mae["per_class"].get(class_name, float('nan'))
        print(f"{class_name:<28}{mse_val:<12.4f}{mae_val:<12.4f}")

    # Simple overall recommendation based on ROC-AUC (threshold-independent)
    print("\n" + "=" * 60)
    if results_mse["roc_auc"] >= results_mae["roc_auc"]:
        print(f"RECOMMENDATION: MSE performs better (ROC-AUC {results_mse['roc_auc']:.4f} vs {results_mae['roc_auc']:.4f})")
    else:
        print(f"RECOMMENDATION: MAE performs better (ROC-AUC {results_mae['roc_auc']:.4f} vs {results_mse['roc_auc']:.4f})")

except Exception as e:
    import traceback
    print("\n\nSCRIPT FAILED WITH AN ERROR:")
    print(str(e))
    traceback.print_exc()

finally:
    _log_file.close()