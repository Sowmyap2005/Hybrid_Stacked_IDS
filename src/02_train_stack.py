# ==========================================================
# PART 2 — HYBRID IDS TRAINING (NO-PCA VERSION)
# RandomForest + XGBoost + Deep Neural Network + Autoencoder
# Stacking Meta Model
#
# CHANGES FROM YOUR ORIGINAL part2_nopca.py:
#   1. DNN training now captures its history (accuracy/loss
#      per epoch) and saves a corrected 10-epoch plot, fixing
#      the reviewer's epoch-count mismatch comment.
#   2. Added validation_split=0.1 to the DNN so the plot shows
#      both training and validation curves.
# Everything else is unchanged from your working no-PCA pipeline.
# ==========================================================
print("REAL SCRIPT STARTED - THIS IS PART 2 WITH DNN PLOT", flush=True)
import joblib
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns

print("Starting Hybrid IDS Training (no-PCA version)")

# ==========================================================
# LOAD PREPROCESSED DATA (the "_nopca" versions)
# ==========================================================
X_train = joblib.load("X_train_nopca.pkl")
X_test = joblib.load("X_test_nopca.pkl")
y_train = joblib.load("y_train_nopca.pkl")
y_test = joblib.load("y_test_nopca.pkl")

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

le = joblib.load("label_encoder_nopca.pkl")
n_classes = len(le.classes_)
print("Classes:", list(le.classes_))
print("n_classes:", n_classes)
print("Feature count (no PCA):", X_train.shape[1])

# ==========================================================
# SPLIT DATA FOR STACKING (same 70/30 as before)
# ==========================================================
X_base_train, X_meta_train, y_base_train, y_meta_train = train_test_split(
    X_train, y_train,
    test_size=0.3,
    random_state=42,
    stratify=y_train
)

# ==========================================================
# RANDOM FOREST
# ==========================================================
print("\nTraining Random Forest")

rf = RandomForestClassifier(
    n_estimators=50,
    max_depth=12,
    n_jobs=1,
    random_state=42
)
rf.fit(X_base_train, y_base_train)

# ==========================================================
# XGBOOST
# ==========================================================
print("\nTraining XGBoost")

xgb = XGBClassifier(
    n_estimators=50,
    max_depth=6,
    learning_rate=0.1,
    tree_method="hist",
    eval_metric="mlogloss",
    n_jobs=1
)
xgb.fit(X_base_train, y_base_train)

# ==========================================================
# DEEP NEURAL NETWORK
# input size is automatic (X_base_train.shape[1])
# ==========================================================
print("\nTraining Deep Neural Network")

y_base_cat = to_categorical(y_base_train, n_classes)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_base_train),
    y=y_base_train
)
class_weights = dict(enumerate(class_weights))

input_dim = X_base_train.shape[1]

dnn_model = Sequential()
dnn_model.add(Input(shape=(input_dim,)))
dnn_model.add(Dense(128, activation="relu"))
dnn_model.add(Dropout(0.3))
dnn_model.add(Dense(64, activation="relu"))
dnn_model.add(Dropout(0.3))
dnn_model.add(Dense(n_classes, activation="softmax"))

dnn_model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# CHANGE 1: capture history, add validation_split so we get
# both training and validation curves for the corrected figure
history = dnn_model.fit(
    X_base_train, y_base_cat,
    epochs=10,
    batch_size=512,
    class_weight=class_weights,
    validation_split=0.1,
    verbose=1
)

# CHANGE 2: plot and save the real, correct 10-epoch training
# history right here, using the actual history object from
# THIS run — replaces the old, mismatched ~29-epoch figure.
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(history.history['accuracy'], label='Training Accuracy', marker='o')
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
axes[0].set_title('DNN Training and Validation Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].set_xticks(range(len(history.history['accuracy'])))
axes[0].set_xticklabels(range(1, len(history.history['accuracy']) + 1))
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['loss'], label='Training Loss', marker='o')
axes[1].plot(history.history['val_loss'], label='Validation Loss', marker='o')
axes[1].set_title('DNN Training and Validation Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_xticks(range(len(history.history['loss'])))
axes[1].set_xticklabels(range(1, len(history.history['loss']) + 1))
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("dnn_training_history_corrected.png", dpi=200, bbox_inches="tight")
print("Saved DNN training history plot to dnn_training_history_corrected.png")
print(f"Epochs shown: {len(history.history['accuracy'])} (matches paper text: 10)")

# ==========================================================
# AUTOENCODER — input size also automatic
# ==========================================================
print("\nTraining Autoencoder for Zero-Day Detection")

benign_class = le.transform(["BENIGN"])[0]
print("Benign class index:", benign_class)

X_train_benign = X_base_train[y_base_train == benign_class]

input_layer = Input(shape=(input_dim,))
encoded = Dense(16, activation='relu')(input_layer)
encoded = Dense(8, activation='relu')(encoded)
decoded = Dense(16, activation='relu')(encoded)
decoded = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

autoencoder.fit(
    X_train_benign, X_train_benign,
    epochs=10,
    batch_size=512,
    shuffle=True,
    validation_split=0.2,
    verbose=1
)

# ==========================================================
# CREATE META FEATURES
# ==========================================================
print("\nCreating Meta Features for Stacking")

rf_meta = rf.predict_proba(X_meta_train)
xgb_meta = xgb.predict_proba(X_meta_train)
dnn_meta = dnn_model.predict(X_meta_train)

recon_meta = autoencoder.predict(X_meta_train)
mse_meta = np.mean(np.power(X_meta_train - recon_meta, 2), axis=1).reshape(-1, 1)

meta_features = np.hstack((rf_meta, xgb_meta, dnn_meta, mse_meta))

# ==========================================================
# META MODEL (STACKING)
# ==========================================================
print("\nTraining Meta Model (Stacking)")

meta_model = LogisticRegression(max_iter=1000, class_weight="balanced")
meta_model.fit(meta_features, y_meta_train)

# ==========================================================
# TEST PREDICTIONS
# ==========================================================
print("\nEvaluating Model")

rf_test = rf.predict_proba(X_test)
xgb_test = xgb.predict_proba(X_test)
dnn_test = dnn_model.predict(X_test)

recon_test = autoencoder.predict(X_test)
mse_test = np.mean(np.power(X_test - recon_test, 2), axis=1).reshape(-1, 1)

meta_test = np.hstack((rf_test, xgb_test, dnn_test, mse_test))
final_pred = meta_model.predict(meta_test)

# ==========================================================
# OVERALL IDS PERFORMANCE
# ==========================================================
print("\n==============================")
print("OVERALL IDS PERFORMANCE (no-PCA version)")
print("==============================")

accuracy = accuracy_score(y_test, final_pred)
precision = precision_score(y_test, final_pred, average="weighted")
recall = recall_score(y_test, final_pred, average="weighted")
f1 = f1_score(y_test, final_pred, average="weighted")
f1_macro = f1_score(y_test, final_pred, average="macro")

print(f"Accuracy    : {accuracy:.4f}")
print(f"Precision   : {precision:.4f}")
print(f"Recall      : {recall:.4f}")
print(f"Weighted F1 : {f1:.4f}")
print(f"Macro F1    : {f1_macro:.4f}")

print("\nPer-class classification report:")
print(classification_report(y_test, final_pred, target_names=[str(c) for c in le.classes_]))

# ==========================================================
# CONFUSION MATRIX
# ==========================================================
cm = confusion_matrix(y_test, final_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, cmap="Blues")
plt.title("Confusion Matrix - IDS (no-PCA version)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix_nopca.png", dpi=200, bbox_inches="tight")
plt.show()

# ==========================================================
# SAVE MODELS
# ==========================================================
joblib.dump(rf, "random_forest_model_nopca.pkl")
joblib.dump(xgb, "xgb_model_nopca.pkl")
joblib.dump(meta_model, "meta_model_nopca.pkl")
dnn_model.save("dnn_model_nopca.keras")
autoencoder.save("autoencoder_model_nopca.keras")

print("\nHybrid IDS Models (no-PCA version) Saved Successfully")