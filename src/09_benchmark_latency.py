# ==========================================================
# INFERENCE LATENCY BENCHMARK (no-PCA version)
# Measures how long the FULL stacked pipeline takes to classify
# flows, at a few different batch sizes. This answers reviewer
# comment 16: "no inference latency, throughput, or memory
# footprint reported."
# ==========================================================

import joblib
import numpy as np
import time
from tensorflow.keras.models import load_model

print("Starting inference latency benchmark...")

# ==========================================================
# LOAD EVERYTHING (same as your anomaly_check_nopca.py)
# ==========================================================
X_test = joblib.load("X_test_nopca.pkl")
y_test = joblib.load("y_test_nopca.pkl")
le = joblib.load("label_encoder_nopca.pkl")

rf = joblib.load("random_forest_model_nopca.pkl")
xgb = joblib.load("xgb_model_nopca.pkl")
meta_model = joblib.load("meta_model_nopca.pkl")
dnn_model = load_model("dnn_model_nopca.keras")
autoencoder = load_model("autoencoder_model_nopca.keras")

def run_full_stack(X_batch):
    """Runs the exact same prediction pipeline as your main
    evaluation code: RF + XGBoost + DNN + Autoencoder -> meta-model."""
    rf_pred = rf.predict_proba(X_batch)
    xgb_pred = xgb.predict_proba(X_batch)
    dnn_pred = dnn_model.predict(X_batch, verbose=0)

    recon = autoencoder.predict(X_batch, verbose=0)
    mse = np.mean(np.power(X_batch - recon, 2), axis=1).reshape(-1, 1)

    meta_features = np.hstack((rf_pred, xgb_pred, dnn_pred, mse))
    final_pred = meta_model.predict(meta_features)
    return final_pred

# ==========================================================
# WARM-UP RUN (not timed) — first predictions are always
# slower due to one-time model initialization overhead. We
# exclude this from the real measurement, same as standard
# ML benchmarking practice.
# ==========================================================
print("Running warm-up pass (not timed)...")
_ = run_full_stack(X_test[:100])

# ==========================================================
# BENCHMARK AT SEVERAL BATCH SIZES
# ==========================================================
batch_sizes = [1, 100, 1000, 10000]
results = []

for batch_size in batch_sizes:
    if batch_size > len(X_test):
        continue

    X_batch = X_test[:batch_size]

    start = time.perf_counter()
    _ = run_full_stack(X_batch)
    elapsed = time.perf_counter() - start

    per_flow_ms = (elapsed / batch_size) * 1000
    throughput = batch_size / elapsed

    print(f"\nBatch size: {batch_size}")
    print(f"  Total time: {elapsed:.4f} sec")
    print(f"  Per-flow latency: {per_flow_ms:.4f} ms")
    print(f"  Throughput: {throughput:.1f} flows/sec")

    results.append({
        "batch_size": batch_size,
        "total_time_sec": elapsed,
        "per_flow_ms": per_flow_ms,
        "throughput_flows_per_sec": throughput
    })

# ==========================================================
# SUMMARY TABLE
# ==========================================================
print("\n" + "=" * 60)
print("LATENCY BENCHMARK SUMMARY")
print("=" * 60)
print(f"{'Batch Size':<15}{'Per-Flow (ms)':<18}{'Throughput (flows/s)':<20}")
print("-" * 60)
for r in results:
    print(f"{r['batch_size']:<15}{r['per_flow_ms']:<18.4f}{r['throughput_flows_per_sec']:<20.1f}")

print("\nNote: this measures CPU inference time for the full 4-model")
print("stacked pipeline (RF + XGBoost + DNN + Autoencoder + meta-model)")
print("as it currently runs, without any deployment-specific optimization")
print("(e.g., model quantization, batching strategy, or GPU acceleration).")