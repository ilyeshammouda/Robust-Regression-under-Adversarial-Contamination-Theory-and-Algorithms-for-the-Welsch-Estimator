import numpy as np
from scipy.optimize import fixed_point, minimize
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from tqdm import tqdm

from algorithms.help_functions import Welsch_tools, score


class LeastAbsoluteDeviationAlgo:
    """
    Estimates beta using the Least Absolute Deviation (L1) loss function.

    Supports two approaches:
        - optimizer_approach: solves the optimization problem directly via scipy solvers.
        - gradient_descent_approach: optimizes the L1 loss using gradient descent.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, sigma: float = 1.0):
        self.X = X
        self.y = y
        self.sigma = sigma
        self.n, self.p = X.shape

    def l1_loss(self, beta: np.ndarray) -> float:
        """Compute the mean absolute error (L1 loss)."""
        return np.mean(np.abs(self.y - self.X @ beta))

    def l1_loss_gradient(self, beta: np.ndarray) -> np.ndarray:
        """Compute the gradient of the L1 loss."""
        residuals = self.y - self.X @ beta
        return -self.X.T @ np.sign(residuals) / self.n

    def optimizer_approach(
        self,
        initial_guess: np.ndarray = None,
        method: str = "BFGS",
        max_iter: int = 10,
    ) -> tuple[np.ndarray, int]:
        """
        Minimize the L1 loss using a scipy solver.

        Returns:
            beta_hat: estimated coefficients.
            n_iterations: number of iterations performed.
        """
        if initial_guess is None:
            initial_guess = np.random.normal(loc=0, scale=1, size=self.p)

        result = minimize(
            self.l1_loss,
            initial_guess,
            method=method,
            options={"maxiter": max_iter},
        )
        return result.x, result.nit

    def gradient_descent_approach(
        self,
        initial_guess: np.ndarray = None,
        learning_rate: float = 0.001,
        iterations: int = 1000,
        decay_rate: float = 0.8,
    ) -> np.ndarray:
        """
        Minimize the L1 loss using gradient descent with exponential learning rate decay.

        Returns:
            beta_hat: estimated coefficients.
        """
        if initial_guess is None:
            initial_guess = np.random.normal(
                loc=0, scale=np.sqrt(2 / (self.p + self.n)), size=self.p
            )

        beta = initial_guess.copy()
        lr = learning_rate

        for i in range(iterations):
            lr = learning_rate * (decay_rate ** i)
            gradient = self.l1_loss_gradient(beta)
            beta -= lr * gradient

        return beta


class WelschAlgo:
    """
    Estimates beta using Welsch loss.

    Supports three approaches:
        - fixed_point_approach: iterative fixed-point method.
        - optimizer_approach: direct optimization via scipy solvers (with L1 warm start).
        - gradient_descent_approach: Nesterov-style gradient descent (with L1 warm start).
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, sigma: float = 1.0):
        self.X = X
        self.y = y
        self.sigma = sigma
        self.n, self.p = X.shape
        self.tools = Welsch_tools(self.X, self.y, self.sigma)

    # ------------------------------------------------------------------ #
    # Fixed-point approach
    # ------------------------------------------------------------------ #

    def fixed_point_approach(
        self,
        tau: float,
        initial_guess: np.ndarray = None,
        maxiter: int = 1000,
    ) -> np.ndarray:
        """
        Find the estimator via fixed-point iteration.

        Returns:
            beta_hat: estimated coefficients.
        """
        if initial_guess is None:
            initial_guess = np.random.normal(
                loc=0, scale=np.sqrt(4 / (self.p + self.n)), size=self.p
            )

        return fixed_point(
            lambda beta: self.tools.function_fixed_point(beta, tau),
            initial_guess,
            maxiter=maxiter,
        )

    # ------------------------------------------------------------------ #
    # Optimizer approach
    # ------------------------------------------------------------------ #

    def optimizer_approach(
        self,
        tau: float,
        initial_guess: np.ndarray = None,
        method: str = "BFGS",
        maxiter: int = 10,
        maxiter_first_stage: int = 100,
    ) -> np.ndarray:
        """
        Minimize the alpha-divergence loss using a scipy solver.

        A first stage uses L1 regression to obtain a warm-start estimate and
        to rescale tau by the estimated residual variance.

        Returns:
            beta_hat: estimated coefficients.
        """
        if initial_guess is None:
            initial_guess = np.random.normal(
                loc=0, scale=np.sqrt(4 / (self.p + self.n)), size=self.p
            )

        # Warm start with L1 regression
        lad = LeastAbsoluteDeviationAlgo(X=self.X, y=self.y)
        initial_guess, _ = lad.optimizer_approach(
            method="BFGS",
            initial_guess=initial_guess,
            max_iter=maxiter_first_stage,
        )

        # Rescale tau using the estimated residual scale
        result = minimize(
            lambda beta: self.tools.Welsch_loss(beta, tau),
            initial_guess,
            method=method,
            tol=1e-50,
            options={"maxiter": maxiter},
        )
        return result.x

    # ------------------------------------------------------------------ #
    # Gradient descent approach
    # ------------------------------------------------------------------ #

    def gradient_descent_approach(
        self,
        tau: float,
        initial_guess: np.ndarray = None,
        beta_star: np.ndarray = None,
        learning_rate: float = 0.01,
        iterations: int = 1000,
        decay_rate: float = 0.8,
        track_score: bool = False,
        score_threshold: float = 0.5,
        skip_warm_start: bool = False,
    ) -> tuple:
        """
        Minimize the alpha-divergence loss using gradient descent with
        exponential learning rate decay.

        Unless `skip_warm_start` is True, a first stage uses L1 optimization
        to bring the initial guess into the basin of attraction.

        Args:
            tau: divergence tuning parameter.
            initial_guess: starting point for optimization.
            beta_star: true coefficients (for tracking convergence on simulated data).
            learning_rate: initial step size.
            iterations: number of gradient descent steps.
            decay_rate: exponential decay factor for the learning rate.
            track_score: if True, record the score at each iteration.
            score_threshold: threshold for the warm-start stopping criterion.
            skip_warm_start: if True, skip the L1 warm-start stage.

        Returns:
            beta_hat: estimated coefficients.
            distance_list (optional): list of (iteration, distance) tuples (only if beta_star is given).
            score_list: list of scores per iteration (empty if track_score is False).
        """
        if initial_guess is None:
            initial_guess = np.random.normal(
                loc=0, scale=np.sqrt(4 / (self.p + self.n)), size=self.p
            )

        distance_list = [] if beta_star is not None else None

        # Warm-start stage: use L1 to reach the basin of attraction
        warm_start_iters = 0
        if not skip_warm_start:
            lad = LeastAbsoluteDeviationAlgo(X=self.X, y=self.y)
            while score(X=self.X, y=self.y, tau=tau, beta=initial_guess) > score_threshold:
                warm_start_iters += 1
                initial_guess, _ = lad.optimizer_approach(
                    method="BFGS", initial_guess=initial_guess, max_iter=1
                )
                if beta_star is not None:
                    distance_list.append(
                        (warm_start_iters, np.linalg.norm(initial_guess - beta_star))
                    )
                if warm_start_iters > 100:
                    break

        # Main gradient descent stage
        score_list = []
        beta = initial_guess.copy()
        lr = learning_rate

        for i in range(iterations):
            gradient = self.tools.gradient_Welsch_loss(beta, tau)
            beta -= lr * gradient
            lr = learning_rate * (decay_rate ** i)

            if beta_star is not None:
                distance_list.append(
                    (warm_start_iters + i, np.linalg.norm(beta - beta_star))
                )
            if track_score:
                score_list.append(score(X=self.X, y=self.y, tau=tau, beta=beta))

        if beta_star is not None:
            return beta, distance_list, score_list
        return beta, score_list

    # ------------------------------------------------------------------ #
    # Cross-validated grid search for tau
    # ------------------------------------------------------------------ #

    def grid_search_cv(
        self,
        tau_values: list[float],
        approach_method: str,
        n_splits: int = 5,
        initial_guess: np.ndarray = None,
        learning_rate: float = 0.01,
        iterations: int = 100,
    ) -> float:
        """
        Select the best tau via K-Fold cross-validation (median MSE).

        Args:
            tau_values: candidate values for tau.
            approach_method: one of 'fixed_point', 'optimizer', or 'gradient_descent'.
            n_splits: number of CV folds.

        Returns:
            best_tau: the tau value with the lowest median validation error.
        """
        kf = KFold(n_splits=n_splits)
        best_tau = None
        best_error = float("inf")

        for tau in tqdm(tau_values, desc="Grid search"):
            fold_errors = []

            for train_idx, val_idx in kf.split(self.X):
                X_train, X_val = self.X[train_idx], self.X[val_idx]
                y_train, y_val = self.y[train_idx], self.y[val_idx]

                # Temporarily swap data for fitting
                original_X, original_y = self.X, self.y
                self.X, self.y = X_train, y_train
                self.n, self.p = X_train.shape
                self.tools = Welsch_tools(self.X, self.y, self.sigma)

                if approach_method == "fixed_point":
                    beta_hat = self.fixed_point_approach(tau)
                elif approach_method == "optimizer":
                    beta_hat = self.optimizer_approach(tau, maxiter=iterations)
                elif approach_method == "gradient_descent":
                    beta_hat, _ = self.gradient_descent_approach(
                        tau,
                        initial_guess=initial_guess,
                        learning_rate=learning_rate,
                        iterations=iterations,
                    )
                else:
                    raise ValueError(f"Unknown approach: {approach_method}")

                # Restore original data
                self.X, self.y = original_X, original_y
                self.n, self.p = self.X.shape
                self.tools = Welsch_tools(self.X, self.y, self.sigma)

                y_pred = X_val @ beta_hat
                fold_errors.append(mean_squared_error(y_val, y_pred))

            median_error = np.median(fold_errors)
            if median_error < best_error:
                best_error = median_error
                best_tau = tau

        return best_tau