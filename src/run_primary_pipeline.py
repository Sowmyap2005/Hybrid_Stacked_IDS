"""Run the primary preprocessing, training, and evaluation workflow.

Run from the repository root:
    python src/run_primary_pipeline.py
"""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STEPS = [
    "src/01_preprocess.py",
    "src/02_train_stack.py",
    "src/03_evaluate_auc.py",
    "src/04_normalized_confusion.py",
    "src/05_plot_roc_curves.py",
]

for step in STEPS:
    print(f"\n{'=' * 72}\nRunning {step}\n{'=' * 72}")
    subprocess.run([sys.executable, str(ROOT / step)], cwd=ROOT, check=True)

print("\nPrimary pipeline completed successfully.")
