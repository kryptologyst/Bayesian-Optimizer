import typer
import sys
from loguru import logger

from .config import settings
from .data import load_data
from .model import BayesianOptimizer
from .visualizer import BOVisualizer

app = typer.Typer(help="Bayesian Optimizer CLI")
logger.remove()
logger.add(sys.stderr, level=settings.log_level)


@app.command()
def optimize(
    dataset: str = typer.Option("wine", help="Dataset: iris, wine, breast_cancer"),
    n_iter: int = typer.Option(30, help="Optimization iterations"),
    visualize: bool = typer.Option(True, help="Generate plots"),
):
    logger.info(f"Running Bayesian optimization on {dataset} ({n_iter} iters)...")
    X, y, _ = load_data(dataset)
    param_space = {
        "n_estimators": [10, 300],
        "max_depth": [1, 30],
        "min_samples_split": [2, 20],
        "min_samples_leaf": [1, 10],
    }
    bo = BayesianOptimizer(n_iter=n_iter)
    results = bo.optimize(X, y, param_space)
    logger.info(f"Best params: {results['best_params']}")
    logger.info(f"Best score: {results['best_score']:.3f}")
    if visualize:
        BOVisualizer.plot_convergence(
            results["history"], save_path=settings.plots_dir / "bo_convergence.png",
        )
    logger.success("Optimization complete!")


if __name__ == "__main__":
    app()
