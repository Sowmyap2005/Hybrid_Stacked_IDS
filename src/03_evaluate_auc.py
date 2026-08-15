# ==========================================================
# STACKED MODEL — ROC-AUC, PR-AUC, BALANCED ACCURACY
# Answers reviewer comments 7, 8, and 9: the manuscript needs
# these metrics reported for the FINAL STACKED MODEL specifically
# (not just the DNN, and not just for the autoencoder's binary
# anomaly check we already did separately).
# ==========================================================

import joblib
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    classification_report
)
from sklearn.preprocessing import label_binarize

print("Computing stacked-model ROC-AUC / PR-AUC / balanced accuracy...")

# ==========================================================
# LOAD EVERYTHING (same as your other _nopca scripts)
# ==========================================================
X_test = joblib.load("X_test_nopca.pkl")
y_test = joblib.load("y_test_nopca.pkl")
le = joblib.load("label_encoder_nopca.pkl")
n_classes = len(le.classes_)

rf = joblib.load("random_forest_model_nopca.pkl")
xgb = joblib.load("xgb_model_nopca.pkl")
meta_model = joblib.load("meta_model_nopca.pkl")
dnn_model = load_model("dnn_model_nopca.keras")
autoencoder = load_model("autoencoder_model_nopca.keras")

# ==========================================================
# REBUILD THE STACKED PIPELINE'S PREDICTIONS
# (identical to your main evaluation code, but this time we
# keep the PROBABILITIES, not just the final predicted class)
# ==========================================================
rf_test = rf.predict_proba(X_test)
xgb_test = xgb.predict_proba(X_test)
dnn_test = dnn_model.predict(X_test, verbose=0)

recon_test = autoencoder.predict(X_test, verbose=0)
mse_test = np.mean(np.power(X_test - recon_test, 2), axis=1).reshape(-1, 1)

meta_test = np.hstack((rf_test, xgb_test, dnn_test, mse_test))

# This is the key addition: predict_proba instead of predict,
# so we get a probability for every class, not just the winner
final_proba = meta_model.predict_proba(meta_test)
final_pred = meta_model.predict(meta_test)

# ==========================================================
# 1) BALANCED ACCURACY (answers comment 8's specific ask)
# This is the average of per-class recall — like macro-F1's
# cousin, but specifically for accuracy rather than F1.
# ==========================================================
bal_acc = balanced_accuracy_score(y_test, final_pred)
print(f"\nBalanced Accuracy: {bal_acc:.4f}")

# ==========================================================
# 2) MULTICLASS ROC-AUC (answers comments 7 and 9)
# One-vs-Rest, both macro (unweighted mean across classes)
# and weighted (accounts for class size) — reviewers asked
# for the classes that matter, not just an average that hides
# rare-class weaknesses.
# ==========================================================
y_test_binarized = label_binarize(y_test, classes=range(n_classes))

roc_auc_macro = roc_auc_score(y_test_binarized, final_proba, average="macro", multi_class="ovr")
roc_auc_weighted = roc_auc_score(y_test_binarized, final_proba, average="weighted", multi_class="ovr")

print(f"ROC-AUC (macro, one-vs-rest):    {roc_auc_macro:.4f}")
print(f"ROC-AUC (weighted, one-vs-rest): {roc_auc_weighted:.4f}")

# ==========================================================
# 3) PER-CLASS ROC-AUC AND PR-AUC
# This is the part that actually matters for comment 9 — a
# single averaged ROC-AUC can look great even when specific
# rare classes are weak. Per-class numbers expose that.
# ==========================================================
print("\nPer-class ROC-AUC and PR-AUC:")
print(f"{'Class':<28}{'ROC-AUC':<12}{'PR-AUC':<12}{'Support':<10}")
print("-" * 62)

per_class_results = []
for i, class_name in enumerate(le.classes_):
    class_roc_auc = roc_auc_score(y_test_binarized[:, i], final_proba[:, i])
    class_pr_auc = average_precision_score(y_test_binarized[:, i], final_proba[:, i])
    support = int(y_test_binarized[:, i].sum())
    print(f"{class_name:<28}{class_roc_auc:<12.4f}{class_pr_auc:<12.4f}{support:<10}")
    per_class_results.append({
        "class": class_name, "roc_auc": class_roc_auc,
        "pr_auc": class_pr_auc, "support": support
    })

# ==========================================================
# SAVE FULL RESULTS TO CSV FOR THE PAPER
# ==========================================================
import csv
with open("stacked_model_auc_metrics.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["class", "roc_auc", "pr_auc", "support"])
    for r in per_class_results:
        writer.writerow([r["class"], f"{r['roc_auc']:.4f}", f"{r['pr_auc']:.4f}", r["support"]])

print("\nSaved full per-class table to stacked_model_auc_metrics.csv")

print("\n" + "=" * 60)
print("SUMMARY (for the paper)")
print("=" * 60)
print(f"Balanced Accuracy:        {bal_acc:.4f}")
print(f"ROC-AUC (macro OvR):      {roc_auc_macro:.4f}")
print(f"ROC-AUC (weighted OvR):   {roc_auc_weighted:.4f}")