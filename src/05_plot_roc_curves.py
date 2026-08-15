# ==========================================================
# CHART: ROC CURVES FOR THE FINAL STACKED MODEL (all 12 classes)
# This is a standalone version of the snippet suggested earlier -
# loads your saved models fresh, so it works even if you're not
# running it right after stacked_auc_mertics.py.
#
# IMPORTANT: tensorflow.keras is imported BEFORE sklearn/xgboost,
# matching the fix for the import-order crash found earlier in
# this project.
# ==========================================================

import joblib
import numpy as np

from tensorflow.keras.models import load_model

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

print("Loading saved models and data...")

X_test = joblib.load("X_test_nopca.pkl")
y_test = joblib.load("y_test_nopca.pkl")
le = joblib.load("label_encoder_nopca.pkl")
n_classes = len(le.classes_)

rf = joblib.load("random_forest_model_nopca.pkl")
xgb = joblib.load("xgb_model_nopca.pkl")
meta_model = joblib.load("meta_model_nopca.pkl")
dnn_model = load_model("dnn_model_nopca.keras")
autoencoder = load_model("autoencoder_model_nopca.keras")

print("Computing stacked model predictions...")
rf_test = rf.predict_proba(X_test)
xgb_test = xgb.predict_proba(X_test)
dnn_test = dnn_model.predict(X_test, verbose=0)
recon_test = autoencoder.predict(X_test, verbose=0)
mse_test = np.mean(np.power(X_test - recon_test, 2), axis=1).reshape(-1, 1)
meta_test = np.hstack((rf_test, xgb_test, dnn_test, mse_test))
final_proba = meta_model.predict_proba(meta_test)

print("Building ROC curves...")
y_test_binarized = label_binarize(y_test, classes=range(n_classes))

# Fix the same latin1 encoding glitch (\x96) in class names
class_labels = [c.replace('\x96', '-').strip() for c in le.classes_]

plt.figure(figsize=(9, 8))
colors = plt.cm.tab20(np.linspace(0, 1, n_classes))

for i, (class_name, color) in enumerate(zip(class_labels, colors)):
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], final_proba[:, i])
    auc = roc_auc_score(y_test_binarized[:, i], final_proba[:, i])
    plt.plot(fpr, tpr, color=color, lw=1.5, label=f"{class_name} (AUC={auc:.3f})")

plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Final Stacked Model (per class, one-vs-rest)")
plt.legend(fontsize=7, loc="lower right", ncol=1)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curves_stacked_nopca.png", dpi=200, bbox_inches="tight")
print("Saved to roc_curves_stacked_nopca.png")
