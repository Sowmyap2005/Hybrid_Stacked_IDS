"""Export the numeric values behind Table 2, Table 3, Fig 5 and Fig 6
from the saved final models and held-out test set.
Run from the folder holding the *_nopca model files and X_test/y_test pickles."""
import os, joblib, numpy as np, pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, f1_score,
    balanced_accuracy_score, confusion_matrix, roc_auc_score, roc_curve, average_precision_score)
from sklearn.preprocessing import label_binarize

OUT = os.environ.get("OUT_DIR", "results_export"); os.makedirs(OUT, exist_ok=True)
X = np.asarray(joblib.load("X_test_nopca.pkl")); y = np.asarray(joblib.load("y_test_nopca.pkl"))
le = joblib.load("label_encoder_nopca.pkl"); names = [str(c).replace("\x96", "-") for c in le.classes_]
n = len(names)
rf = joblib.load("random_forest_model_nopca.pkl"); xgb = joblib.load("xgb_model_nopca.pkl")
meta = joblib.load("meta_model_nopca.pkl")
dnn = load_model("dnn_model_nopca.keras"); ae = load_model("autoencoder_model_nopca.keras")

p_rf = rf.predict_proba(X); p_xgb = xgb.predict_proba(X); p_dnn = dnn.predict(X, verbose=0)
mse = np.mean((X - ae.predict(X, verbose=0)) ** 2, axis=1)
M = np.hstack((p_rf, p_xgb, p_dnn, mse.reshape(-1, 1)))
proba = meta.predict_proba(M); pred = meta.predict(M)

def summary(name, yp):
    p, r, f, _ = precision_recall_fscore_support(y, yp, average="weighted", zero_division=0)
    return dict(model=name, accuracy=accuracy_score(y, yp), weighted_precision=p, weighted_recall=r,
                weighted_f1=f, macro_f1=f1_score(y, yp, average="macro"),
                balanced_accuracy=balanced_accuracy_score(y, yp), n_test=len(y))
Yb = label_binarize(y, classes=range(n))
rows = [summary("Stacked (final)", pred), summary("Random Forest", p_rf.argmax(1)),
        summary("XGBoost", p_xgb.argmax(1)), summary("DNN", p_dnn.argmax(1))]
rows[0]["roc_auc_macro_ovr"] = roc_auc_score(Yb, proba, average="macro")
rows[0]["roc_auc_weighted_ovr"] = roc_auc_score(Yb, proba, average="weighted")
pd.DataFrame(rows).round(4).to_csv(f"{OUT}/overall_and_base_model_metrics.csv", index=False)

p, r, f, s = precision_recall_fscore_support(y, pred, labels=range(n), zero_division=0)
pd.DataFrame(dict(class_name=names, precision=p, recall=r, f1=f, support=s,
    roc_auc=[roc_auc_score(Yb[:, i], proba[:, i]) for i in range(n)],
    pr_auc=[average_precision_score(Yb[:, i], proba[:, i]) for i in range(n)])
    ).round(4).to_csv(f"{OUT}/per_class_metrics_stacked.csv", index=False)

cm = confusion_matrix(y, pred, labels=range(n))
pd.DataFrame(cm, index=names, columns=names).to_csv(f"{OUT}/confusion_matrix_counts.csv", index_label="true\\predicted")
pd.DataFrame(cm / cm.sum(1, keepdims=True), index=names, columns=names).round(4).to_csv(
    f"{OUT}/confusion_matrix_row_normalized.csv", index_label="true\\predicted")

roc = []
for i in range(n):
    fpr, tpr, thr = roc_curve(Yb[:, i], proba[:, i])
    roc.append(pd.DataFrame(dict(class_name=names[i], fpr=fpr, tpr=tpr, threshold=thr)))
pd.concat(roc).to_csv(f"{OUT}/roc_curve_points_stacked.csv", index=False)

b = list(le.classes_).index("BENIGN"); mask = y == b
flag = pd.Series(pred[mask][pred[mask] != b]).map(dict(enumerate(names))).value_counts()
pd.DataFrame({"benign_test_flows": [int(mask.sum())], "flagged_non_benign": [int((pred[mask] != b).sum())]}
    ).to_csv(f"{OUT}/benign_flows_flagged_summary.csv", index=False)
flag.rename_axis("predicted_class").rename("count").to_csv(f"{OUT}/benign_flows_flagged_by_class.csv")

pr = pd.DataFrame(proba, columns=[f"p_{c}" for c in names]).round(6)
pr.insert(0, "predicted", [names[k] for k in pred]); pr.insert(0, "true", [names[k] for k in y])
pr.insert(len(pr.columns), "ae_reconstruction_mse", mse.round(6))
pr.to_csv(f"{OUT}/test_set_predictions_stacked.csv.gz", index=False, compression="gzip")
print("Done:", sorted(os.listdir(OUT)))
