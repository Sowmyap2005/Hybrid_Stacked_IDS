# ==========================================================
# EXACT SPLIT EXPORT + REBUILD
#
# Re-runs the SAME sampling / cleaning / splitting as
# src/01_preprocess.py (seed 42) but keeps track of where each
# row came from (source CSV file + row number in that file).
#
# Mode 1 (default) - EXPORT:
#   python src/17_export_split_ids.py
#   -> results/splits/test_ids.csv          (113,095 rows)
#   -> results/splits/train_reduced_ids.csv (90,475 rows, before SMOTE)
#   -> results/splits/split_manifest.json   (file order + SHA-256 of each CSV)
#   and checks the test labels against
#   results/metrics/test_set_predictions_stacked.csv.gz
#
# Mode 2 - REBUILD (anyone with the raw CIC-IDS-2017 CSVs):
#   python src/17_export_split_ids.py --rebuild
#   -> results/splits/test_set_raw.csv  (exact 113,095 raw rows, same order)
#   -> results/splits/train_reduced_raw.csv
#
# Set CICIDS2017_DIR to the folder with the 8 *.pcap_ISCX.csv files.
# ==========================================================
import os, sys, glob, json, hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
OUT = os.path.join(ROOT, "results", "splits")
os.makedirs(OUT, exist_ok=True)
data_dir = os.environ.get("CICIDS2017_DIR", os.path.join(ROOT, "data", "TrafficLabelling"))


def norm(s):
    # "Web Attack \x96 XSS" / "Web Attack – XSS" -> "Web Attack - XSS"
    return " ".join(str(s).replace("\x96", "-").replace("\u2013", "-").split())


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------- REBUILD MODE ----------------
if "--rebuild" in sys.argv:
    for name in ["test", "train_reduced"]:
        ids = pd.read_csv(os.path.join(OUT, f"{name}_ids.csv"))
        parts = []
        for fname, grp in ids.groupby("source_file", sort=False):
            raw = pd.read_csv(os.path.join(data_dir, fname), encoding="latin1", low_memory=False)
            raw.columns = raw.columns.str.strip()
            sel = raw.loc[grp["source_row"].values].copy()
            sel.insert(0, "split_position", grp["split_position"].values)
            sel.insert(1, "source_file", fname)
            sel.insert(2, "source_row", grp["source_row"].values)
            parts.append(sel)
        out = pd.concat(parts).sort_values("split_position")
        assert (out["Label"].map(norm).values == ids["label"].values).all(), "label mismatch"
        out.to_csv(os.path.join(OUT, f"{name}_raw.csv"), index=False)
        print(f"{name}: rebuilt {len(out):,} rows -> results/splits/{name}_raw.csv")
    sys.exit(0)

# ---------------- EXPORT MODE ----------------
# NOTE: glob order must match the order used in 01_preprocess.py
# (01 uses plain glob.glob, so we do the same and record the order).
csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
print("CSV files found:", len(csv_files))
if not csv_files:
    sys.exit("No CSVs found - set CICIDS2017_DIR")

dfs = []
for f in csv_files:
    df = pd.read_csv(f, encoding="latin1", low_memory=False)
    df = df.sample(frac=0.2, random_state=42)              # same as 01
    df["__source_file"] = os.path.basename(f)
    df["__source_row"] = df.index                            # row number in original file (0-based, after header)
    dfs.append(df)
data = pd.concat(dfs, ignore_index=True)
print("After sampling:", data.shape[0])                      # expect 623,870

data.columns = data.columns.str.strip()
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)
print("After cleaning:", data.shape[0])                      # expect 565,487

counts = data["Label"].value_counts()
data = data[data["Label"].isin(counts[counts >= 10].index)].reset_index(drop=True)
print("After rare-class removal:", data.shape[0])            # expect 565,474

y = data["Label"].values
pos = np.arange(len(data))

# train_test_split / resample depend only on (n, y, random_state),
# so splitting row positions gives the SAME partition as 01_preprocess.py
pos_tr, pos_te, y_tr, y_te = train_test_split(pos, y, test_size=0.2, random_state=42, stratify=y)
pos_tr_red, y_tr_red = resample(pos_tr, y_tr, replace=False, n_samples=int(0.2 * len(pos_tr)),
                                random_state=42, stratify=y_tr)
print("Test:", len(pos_te), " Reduced train:", len(pos_tr_red))   # expect 113,095 / 90,475


def export(p, name):
    d = pd.DataFrame({
        "split_position": np.arange(len(p)),
        "source_file": data.loc[p, "__source_file"].values,
        "source_row": data.loc[p, "__source_row"].values,
        "label": [norm(v) for v in data.loc[p, "Label"].values],
    })
    d.to_csv(os.path.join(OUT, f"{name}_ids.csv"), index=False)
    return d

test_ids = export(pos_te, "test")
export(pos_tr_red, "train_reduced")

# ---------- verify against the saved test predictions ----------
pred_path = os.path.join(ROOT, "results", "metrics", "test_set_predictions_stacked.csv.gz")
if os.path.exists(pred_path):
    saved = pd.read_csv(pred_path)["true"].map(norm).values
    same = (saved == test_ids["label"].values)
    print(f"Label match with saved predictions: {same.mean()*100:.4f}% "
          f"({same.sum():,}/{len(same):,})")
    print("EXACT MATCH" if same.all() else "MISMATCH - file order or data differs from the original run")

json.dump({
    "seed": 42,
    "glob_order_used": [os.path.basename(f) for f in csv_files],
    "sha256": {os.path.basename(f): sha256(f) for f in csv_files},
    "n_test": int(len(pos_te)),
    "n_train_reduced": int(len(pos_tr_red)),
    "source_row_meaning": "0-based data row index in the original CSV (header excluded), as read by pandas.read_csv(encoding='latin1')",
}, open(os.path.join(OUT, "split_manifest.json"), "w"), indent=2)
print("Saved results/splits/test_ids.csv, train_reduced_ids.csv, split_manifest.json")
