# ==========================================================
# SHAP ANALYSIS — NO-PCA VERSION
# Computes SHAP values on the XGBoost base classifier trained
# on the full, non-PCA feature set, so results map directly to
# real, named features instead of abstract PCA components.
# ==========================================================

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

print("Starting SHAP analysis (no-PCA)...")

# ==========================================================
# LOAD MODEL + DATA
# ==========================================================
xgb = joblib.load("xgb_model_nopca.pkl")
X_test = joblib.load("X_test_nopca.pkl")
y_test = joblib.load("y_test_nopca.pkl")
le = joblib.load("label_encoder_nopca.pkl")

# ==========================================================
# IMPORTANT: SHAP needs real column/feature names to be useful.
# X_test_nopca.pkl is a plain numpy array (StandardScaler strips
# column names), so we need the ORIGINAL feature name list from
# your part1_nopca.py run. If you still have the original CSV
# columns available, list them here in the SAME ORDER they were
# in when the model was trained (i.e., after dropping Flow ID,
# Source IP, Destination IP, Timestamp, and Label).
#
# EASIEST FIX: re-run this small snippet once inside part1_nopca.py
# right after "X = data.drop("Label", axis=1)" to save the names:
#     joblib.dump(list(X.columns), "feature_names_nopca.pkl")
# Then this script can load them properly below.
# ==========================================================

try:
    feature_names = joblib.load("feature_names_nopca.pkl")
    print(f"Loaded {len(feature_names)} real feature names.")
except FileNotFoundError:
    print("WARNING: feature_names_nopca.pkl not found.")
    print("Falling back to generic Feature_0, Feature_1, ... names.")
    print("To get REAL feature names, add this line to part1_nopca.py")
    print('right after `X = data.drop("Label", axis=1)`:')
    print('    joblib.dump(list(X.columns), "feature_names_nopca.pkl")')
    print("Then rerun part1_nopca.py once, and rerun this script.\n")
    feature_names = [f"Feature_{i}" for i in range(X_test.shape[1])]

X_test_df = pd.DataFrame(X_test, columns=feature_names)

# ==========================================================
# SHAP — use a sample of the test set for speed (SHAP on the
# full 113,095-row test set would be very slow). 2,000 random
# rows gives a stable, representative picture.
# ==========================================================
sample_size = min(2000, len(X_test_df))
X_sample = X_test_df.sample(n=sample_size, random_state=42)

print(f"Computing SHAP values on a sample of {sample_size} test rows...")
explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_sample)

# For multiclass XGBoost, shap_values can come back in different
# shapes depending on your installed SHAP version:
#   - older versions: a LIST of arrays, one per class, each (n_samples, n_features)
#   - newer versions: a single ndarray shaped (n_samples, n_features, n_classes)
#   - binary/regression case: a single ndarray shaped (n_samples, n_features)
# We handle all three and collapse down to one 1-D importance score per feature.
if isinstance(shap_values, list):
    # older API: list of per-class arrays
    mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
    # newer API: (n_samples, n_features, n_classes)
    mean_abs_shap = np.abs(shap_values).mean(axis=0).mean(axis=-1)
else:
    # 2-D case: (n_samples, n_features)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

mean_abs_shap = np.ravel(mean_abs_shap)  # safety net: force 1-D no matter what

importance_df = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": mean_abs_shap
}).sort_values("mean_abs_shap", ascending=False)

print("\nTop 15 features by mean absolute SHAP value (global importance):")
print(importance_df.head(15).to_string(index=False))

importance_df.to_csv("shap_feature_importance_nopca.csv", index=False)
print("\nFull feature importance table saved to shap_feature_importance_nopca.csv")

# ==========================================================
# SUMMARY PLOT (saved as an image for the paper)
# ==========================================================
try:
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values, X_sample, plot_type="bar",
                           class_names=list(le.classes_), show=False)
    elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
        # newer API: average across the class axis first, so summary_plot
        # gets a plain (n_samples, n_features) array it understands
        shap_values_2d = np.abs(shap_values).mean(axis=-1)
        shap.summary_plot(shap_values_2d, X_sample, plot_type="bar", show=False)
    else:
        shap.summary_plot(shap_values, X_sample, show=False)
except Exception as e:
    print(f"Note: summary plot skipped due to: {e}")
    print("This doesn't affect the feature importance table above, which is the main result needed.")

plt.tight_layout()
plt.savefig("shap_summary_nopca.png", dpi=200, bbox_inches="tight")
print("Summary plot saved to shap_summary_nopca.png")