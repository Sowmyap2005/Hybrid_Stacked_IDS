# ==========================================================
# FULL COMPONENT ABLATION: STACK MINUS ONE COMPONENT AT A TIME
# Tests whether EACH of the 4 components (RF, XGBoost, DNN,
# Autoencoder) actually contributes to the final stacked model,
# by removing one at a time and retraining just the meta-model.
#
# Reuses your already-trained RF, XGBoost, DNN, and Autoencoder
# — no retraining of those needed. Only small new meta-models
# are trained (fast), one per combination.
# ==========================================================

import joblib
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

print("Starting full component ablation study...")

# ==========================================================
# LOAD DATA AND ALL ALREADY-TRAINED BASE MODELS
# ==========================================================
X_train = joblib.load("X_train_nopca.pkl")
y_train = joblib.load("y_train_nopca.pkl")
X_test = joblib.load("X_test_nopca.pkl")
y_test = joblib.load("y_test_nopca.pkl")
le = joblib.load("label_encoder_nopca.pkl")

rf = joblib.load("random_forest_model_nopca.pkl")
xgb = joblib.load("xgb_model_nopca.pkl")
dnn_model = load_model("dnn_model_nopca.keras")
autoencoder = load_model("autoencoder_model_nopca.keras")

# ==========================================================
# RECREATE THE SAME 70/30 BASE/META SPLIT as your main pipeline
# ==========================================================
X_base_train, X_meta_train, y_base_train, y_meta_train = train_test_split(
    X_train, y_train, test_size=0.3, random_state=42, stratify=y_train
)

# ==========================================================
# COMPUTE EACH COMPONENT'S OUTPUT ONCE (reused across all
# combinations below, so we don't repeat this 4 times)
# ==========================================================
print("Computing each component's predictions on meta-train and test sets...")

rf_meta = rf.predict_proba(X_meta_train)
xgb_meta = xgb.predict_proba(X_meta_train)
dnn_meta = dnn_model.predict(X_meta_train, verbose=0)
recon_meta = autoencoder.predict(X_meta_train, verbose=0)
mse_meta = np.mean(np.power(X_meta_train - recon_meta, 2), axis=1).reshape(-1, 1)

rf_test = rf.predict_proba(X_test)
xgb_test = xgb.predict_proba(X_test)
dnn_test = dnn_model.predict(X_test, verbose=0)
recon_test = autoencoder.predict(X_test, verbose=0)
mse_test = np.mean(np.power(X_test - recon_test, 2), axis=1).reshape(-1, 1)

print("Done. Now training one meta-model per combination...\n")

# ==========================================================
# HELPER: train a meta-model on a chosen subset of components,
# evaluate on test, print and return results
# ==========================================================
def run_combination(name, train_parts, test_parts):
    meta_train = np.hstack(train_parts)
    meta_test = np.hstack(test_parts)

    meta_model = LogisticRegression(max_iter=1000, class_weight="balanced")
    meta_model.fit(meta_train, y_meta_train)
    pred = meta_model.predict(meta_test)

    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, pred, average="weighted", zero_division=0)
    f1w = f1_score(y_test, pred, average="weighted", zero_division=0)
    f1m = f1_score(y_test, pred, average="macro", zero_division=0)

    print(f"{name:<30} Acc={acc:.4f}  W-F1={f1w:.4f}  Macro-F1={f1m:.4f}")

    return {"name": name, "accuracy": acc, "precision": prec,
            "recall": rec, "weighted_f1": f1w, "macro_f1": f1m}

# ==========================================================
# RUN ALL COMBINATIONS
# ==========================================================
results = []

# Full stack (all 4) — the baseline to compare everything against
results.append(run_combination(
    "Full stack (all 4)",
    [rf_meta, xgb_meta, dnn_meta, mse_meta],
    [rf_test, xgb_test, dnn_test, mse_test]
))

# Stack minus Random Forest
results.append(run_combination(
    "Stack minus RF",
    [xgb_meta, dnn_meta, mse_meta],
    [xgb_test, dnn_test, mse_test]
))

# Stack minus XGBoost
results.append(run_combination(
    "Stack minus XGBoost",
    [rf_meta, dnn_meta, mse_meta],
    [rf_test, dnn_test, mse_test]
))

# Stack minus DNN
results.append(run_combination(
    "Stack minus DNN",
    [rf_meta, xgb_meta, mse_meta],
    [rf_test, xgb_test, mse_test]
))

# Stack minus Autoencoder
results.append(run_combination(
    "Stack minus Autoencoder",
    [rf_meta, xgb_meta, dnn_meta],
    [rf_test, xgb_test, dnn_test]
))

# ==========================================================
# FINAL SUMMARY TABLE
# ==========================================================
print("\n" + "=" * 70)
print("FULL COMPONENT ABLATION — SUMMARY TABLE")
print("=" * 70)
print(f"{'Configuration':<30}{'Accuracy':<12}{'Weighted F1':<14}{'Macro F1':<12}")
print("-" * 70)
for r in results:
    print(f"{r['name']:<30}{r['accuracy']:.4f}      {r['weighted_f1']:.4f}        {r['macro_f1']:.4f}")

print("\nA drop in Macro-F1 when a component is removed means that")
print("component genuinely contributes to detecting rare/minority classes.")
print("A negligible change means that component adds little beyond the others.")