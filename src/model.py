import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.stats import norm
from loguru import logger
from typing import Callable


class BayesianOptimizer:
    def __init__(
        self,
        n_iter: int = 30,
        n_initial: int = 5,
        random_state: int = 42,
    ):
        self.n_iter = n_iter
        self.n_initial = n_initial
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.scaler = StandardScaler()
        self.gp = None
        self.X_observed_: list = []
        self.y_observed_: list = []
        self.best_params_: dict = {}
        self.best_score_: float = 0.0

    def optimize(
        self, X: np.ndarray, y: np.ndarray,
        param_space: dict, objective: Callable = None,
    ) -> dict:
        X_scaled = self.scaler.fit_transform(X)
        if objective is None:
            objective = self._default_objective(X_scaled, y)
        keys = list(param_space.keys())
        bounds = np.array([[min(v), max(v)] for v in param_space.values()])
        for _ in range(self.n_initial):
            point = self.rng.uniform(bounds[:, 0], bounds[:, 1])
            score = objective(self._to_params(point, keys))
            self.X_observed_.append(point)
            self.y_observed_.append(score)
        for i in range(self.n_iter):
            X_obs = np.array(self.X_observed_)
            y_obs = np.array(self.y_observed_)
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
            self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=self.random_state)
            self.gp.fit(X_obs, y_obs)
            next_point = self._acquisition(X_obs, y_obs, bounds)
            score = objective(self._to_params(next_point, keys))
            self.X_observed_.append(next_point)
            self.y_observed_.append(score)
            if score > self.best_score_:
                self.best_score_ = score
                self.best_params_ = self._to_params(next_point, keys)
            if (i + 1) % 10 == 0:
                logger.info(f"Iter {i+1}/{self.n_iter}: best={self.best_score_:.3f}")
        best_idx = np.argmax(self.y_observed_)
        self.best_params_ = self._to_params(np.array(self.X_observed_[best_idx]), keys)
        self.best_score_ = float(self.y_observed_[best_idx])
        logger.info(f"Best: {self.best_params_} → {self.best_score_:.3f}")
        return {
            "best_params": self.best_params_,
            "best_score": self.best_score_,
            "history": {
                "params": [self._to_params(np.array(p), keys) for p in self.X_observed_],
                "scores": [float(s) for s in self.y_observed_],
            },
        }

    def _acquisition(self, X_obs, y_obs, bounds, xi=0.01):
        best_y = np.max(y_obs)
        n_candidates = 1000
        candidates = self.rng.uniform(bounds[:, 0], bounds[:, 1], (n_candidates, bounds.shape[0]))
        mu, sigma = self.gp.predict(candidates, return_std=True)
        sigma = np.maximum(sigma, 1e-9)
        improvement = mu - best_y - xi
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        return candidates[np.argmax(ei)]

    def _to_params(self, point, keys):
        return {k: float(v) if isinstance(v, (np.floating, float)) else int(v) for k, v in zip(keys, point)}

    def _default_objective(self, X, y):
        def obj(params):
            n_est = int(params.get("n_estimators", 100))
            max_d = params.get("max_depth", 10)
            model = RandomForestClassifier(
                n_estimators=max(10, n_est),
                max_depth=int(max_d) if max_d > 0 else None,
                random_state=42, n_jobs=-1,
            )
            scores = cross_val_score(model, X, y, cv=3, scoring="accuracy")
            return float(np.mean(scores))
        return obj
