import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
from loguru import logger


class BOVisualizer:
    @staticmethod
    def plot_convergence(history: dict, save_path=None):
        scores = history["scores"]
        best_so_far = np.maximum.accumulate(scores)
        plt.figure(figsize=(10, 5))
        plt.plot(scores, "o-", alpha=0.4, markersize=4, label="Observed", color="steelblue")
        plt.plot(best_so_far, "r-", linewidth=2, label="Best so far")
        plt.xlabel("Iteration"); plt.ylabel("Score")
        plt.title("Bayesian Optimization Convergence"); plt.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
