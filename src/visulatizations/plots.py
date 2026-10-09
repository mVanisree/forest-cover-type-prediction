"""Create project charts from generated CSV result files."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def create_plots(results_dir="results"):
    results_dir = Path(results_dir)
    model_path = results_dir / "model_comparison_regenerated.csv"
    size_path = results_dir / "training_size_regenerated.csv"
    if model_path.exists():
        df = pd.read_csv(model_path)
        ax = df.set_index("Model")[["Accuracy", "Macro F1", "Weighted F1"]].plot(kind="bar", ylim=(0,1), figsize=(9,5))
        ax.set_ylabel("Score"); ax.set_title("Ensemble Model Comparison")
        plt.tight_layout(); plt.savefig(results_dir / "model_comparison.png", dpi=160); plt.close()
    if size_path.exists():
        df = pd.read_csv(size_path)
        plt.figure(figsize=(8,5))
        plt.plot(df["Training Samples"], df["Accuracy"], marker="o", label="Accuracy")
        plt.plot(df["Training Samples"], df["Macro F1"], marker="s", label="Macro F1")
        plt.xlabel("Training samples"); plt.ylabel("Score"); plt.title("Performance vs Training Data Size")
        plt.legend(); plt.tight_layout(); plt.savefig(results_dir / "training_size_performance.png", dpi=160); plt.close()
    return sorted(p.name for p in results_dir.glob("*.png"))
