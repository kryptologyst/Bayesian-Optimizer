import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import BayesianOptimizer
from src.data import load_data


class TestBayesianOptimizer:
    @pytest.fixture
    def data(self):
        return load_data("wine")

    def test_optimize(self, data):
        X, y, _ = data
        param_space = {"n_estimators": [10, 200], "max_depth": [1, 20]}
        bo = BayesianOptimizer(n_iter=10, n_initial=3)
        results = bo.optimize(X, y, param_space)
        assert results["best_score"] > 0
        assert len(results["best_params"]) == 2
        assert len(results["history"]["scores"]) == 13

    def test_convergence(self, data):
        X, y, _ = data
        param_space = {"n_estimators": [10, 200], "max_depth": [1, 20]}
        bo = BayesianOptimizer(n_iter=15, n_initial=3)
        results = bo.optimize(X, y, param_space)
        scores = results["history"]["scores"]
        best_so_far = np.maximum.accumulate(scores)
        assert best_so_far[-1] >= best_so_far[0]
