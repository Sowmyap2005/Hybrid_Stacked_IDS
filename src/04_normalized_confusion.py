# ==========================================================
# NORMALIZED CONFUSION MATRIX (row-normalized, i.e. recall per cell)
# Fixes the readability problem in the raw-count version: BENIGN's
# huge count (90,844) dominates the color scale so badly that every
# other class's diagonal looks nearly white. Row-normalizing shows
# each class's OWN performance on a 0-1 scale, so every class is
# equally visible regardless of how many samples it has.
# ==========================================================

print("STEP 1: Script started.", flush=True)

import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
print("STEP 2a: Basic imports OK.", flush=True)

# TensorFlow imported BEFORE sklearn.metrics, to avoid the same
# import-order crash found earlier in this project.
from tensorflow.keras.models import load_model
print("STEP 2b: TensorFlow imported OK.", flush=True)

from sklearn.metrics import confusion_matrix
print("STEP 2: All imports OK.", flush=True)

X_test = joblib.load("X_test_nopca.pkl")
y_test = joblib.load("y_test_nopca.pkl")
le = joblib.load("label_encoder_nopca.pkl")
print("STEP 3: Data files loaded OK.", flush=True)

rf = joblib.load("random_forest_model_nopca.pkl")
xgb = joblib.load("xgb_model_nopca.pkl")
meta_model = joblib.load("meta_model_nopca.pkl")
print("STEP 4: RF, XGBoost, meta-model loaded OK.", flush=True)

print("STEP 5: About to load DNN model...", flush=True)
dnn_model = load_model("dnn_model_nopca.keras")
print("STEP 6: DNN model loaded OK.", flush=True)

print("STEP 7: About to load Autoencoder model...", flush=True)
autoencoder = load_model("autoencoder_model_nopca.keras")
print("STEP 8: Autoencoder model loaded OK.", flush=True)

rf_test = rf.predict_proba(X_test)
print("STEP 9: RF predictions done.", flush=True)
xgb_test = xgb.predict_proba(X_test)
print("STEP 10: XGBoost predictions done.", flush=True)
dnn_test = dnn_model.predict(X_test, verbose=0)
print("STEP 11: DNN predictions done.", flush=True)
recon_test = autoencoder.predict(X_test, verbose=0)
print("STEP 12: Autoencoder predictions done.", flush=True)
mse_test = np.mean(np.power(X_test - recon_test, 2), axis=1).reshape(-1, 1)
meta_test = np.hstack((rf_test, xgb_test, dnn_test, mse_test))
final_pred = meta_model.predict(meta_test)
print("STEP 13: Final stacked predictions done.", flush=True)

# ==========================================================
# NORMALIZE: divide each row by its own total, so every row
# (true class) sums to 1.0 - shows recall-per-cell, not raw counts
# ==========================================================
cm = confusion_matrix(y_test, final_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
print("STEP 14: Confusion matrix computed. Building plot...", flush=True)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm_normalized,
    annot=True, fmt=".2f",       # show the actual proportion in each cell
    cmap="Blues",
    xticklabels=le.classes_,
    yticklabels=le.classes_,
    cbar_kws={'label': 'Proportion of true class'}
)
plt.title("Normalized Confusion Matrix - Stacked Model (no-PCA)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("confusion_matrix_normalized.png", dpi=200, bbox_inches="tight")
print("STEP 15: SUCCESS. Saved to confusion_matrix_normalized.png", flush=True)