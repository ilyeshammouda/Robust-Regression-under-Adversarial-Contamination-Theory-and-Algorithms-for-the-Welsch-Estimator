import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from tqdm import tqdm


class HuberAlgo:
    """
    Estimates beta using the Huber loss function.

    Supports two approaches:
        - optimizer_approach: solves the optimization problem directly via scipy solvers.
        - gradient_descent_approach: optimizes via gradient descent with learning rate decay.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, sigma: float = 1.0):
        self.X = X
        self.y = y
        self.sigma = sigma
        self.n, self.p = X.shape

    # ------------------------------------------------------------------ #
    # Loss and gradient
    # ------------------------------------------------------------------ #

    def huber_loss(self, u: np.ndarray, gamma: float) -> np.ndarray:
        """Element-wise Huber loss."""
        return np.where(
            np.abs(u) <= gamma,
            0.5 * u ** 2,
            gamma * (np.abs(u) - 0.5 * gamma),
        )

    def huber_loss_gradient(self, u: np.ndarray, gamma: float) -> np.ndarray:
        """Element-wise gradient of the Huber loss."""
        return np.where(np.abs(u) <= gamma, u, gamma * np.sign(u))

    def huber_loss_regression(self, beta: np.ndarray, gamma: float) -> float:
        """Mean Huber loss over all residuals."""
        residuals = self.y - self.X @ beta
        return self.huber_loss(residuals, gamma).mean()

    def huber_loss_regression_gradient(
        self, beta: np.ndarray, gamma: float
    ) -> np.ndarray:
        """Gradient of the mean Huber regression loss w.r.t. beta."""
        residuals = self.y - self.X @ beta
        return -self.X.T @ self.huber_loss_gradient(residuals, gamma) / self.n

    # ------------------------------------------------------------------ #
    # Optimizer approach
    # ------------------------------------------------------------------ #

    def optimizer_approach(
        self,
        gamma: float,
        initial_guess: np.ndarray = None,
        method: str = "L-BFGS-B",
        max_iter: int = 10,
    ) -> tuple[np.ndarray, int]:
        """
        Minimize the Huber regression loss using a scipy solver.

        Returns:
            beta_hat: estimated coefficients.
            n_iterations: number of iterations performed.
        """
        if initial_guess is None:
            initial_guess = np.zeros(self.p)

        result = minimize(
            lambda beta: self.huber_loss_regression(beta, gamma),
            initial_guess,
            method=method,
            tol=1e-50,
            options={"maxiter": max_iter},
        )
        return result.x, result.nit

    # ------------------------------------------------------------------ #
    # Gradient descent approach
    # ------------------------------------------------------------------ #

    def gradient_descent_approach(
        self,
        gamma: float,
        initial_guess: np.ndarray = None,
        beta_star: np.ndarray = None,
        learning_rate: float = 0.01,
        iterations: int = 1000,
        decay_rate: float = 0.8,
    ) -> tuple:
        """
        Minimize the Huber regression loss using gradient descent with
        exponential learning rate decay.

        Args:
            gamma: Huber threshold parameter.
            initial_guess: starting point for optimization.
            beta_star: true coefficients (for tracking convergence on simulated data).
            learning_rate: initial step size.
            iterations: number of gradient descent steps.
            decay_rate: exponential decay factor for the learning rate.

        Returns:
            If beta_star is provided: (beta_hat, convergence_distances).
            Otherwise: beta_hat.
        """
        if initial_guess is None:
            initial_guess = np.random.normal(
                loc=0, scale=np.sqrt(4 / (self.p + self.n)), size=self.p
            )

        beta = initial_guess.copy()
        lr = learning_rate
        convergence = [np.linalg.norm(beta - beta_star)] if beta_star is not None else None

        for i in range(iterations):
            gradient = self.huber_loss_regression_gradient(beta, gamma)
            beta -= lr * gradient
            lr = learning_rate * (decay_rate ** i)

            if convergence is not None:
                convergence.append(np.linalg.norm(beta - beta_star))

        if convergence is not None:
            return beta, convergence
        return beta

    # ------------------------------------------------------------------ #
    # Cross-validated grid search for gamma
    # ------------------------------------------------------------------ #

    def grid_search_cv(
        self,
        gamma_values: list[float],
        approach_method: str,
        n_splits: int = 5,
        initial_guess: np.ndarray = None,
        learning_rate: float = 0.01,
        iterations: int = 100,
    ) -> float:
        """
        Select the best gamma via K-Fold cross-validation (median MSE).

        Args:
            gamma_values: candidate values for gamma.
            approach_method: one of 'optimizer' or 'gradient_descent'.
            n_splits: number of CV folds.

        Returns:
            best_gamma: the gamma value with the lowest median validation error.
        """
        kf = KFold(n_splits=n_splits)
        best_gamma = None
        best_error = float("inf")

        for gamma in tqdm(gamma_values, desc="Grid search"):
            fold_errors = []

            for train_idx, val_idx in kf.split(self.X):
                X_train, X_val = self.X[train_idx], self.X[val_idx]
                y_train, y_val = self.y[train_idx], self.y[val_idx]

                # Temporarily swap data for fitting
                original_X, original_y = self.X, self.y
                self.X, self.y = X_train, y_train
                self.n, self.p = X_train.shape

                if approach_method == "optimizer":
                    beta_hat, _ = self.optimizer_approach(gamma, max_iter=iterations)
                elif approach_method == "gradient_descent":
                    beta_hat = self.gradient_descent_approach(
                        gamma,
                        initial_guess=initial_guess,
                        learning_rate=learning_rate,
                        iterations=iterations,
                    )
                else:
                    raise ValueError(f"Unknown approach: {approach_method}")

                # Restore original data
                self.X, self.y = original_X, original_y
                self.n, self.p = self.X.shape

                y_pred = X_val @ beta_hat
                fold_errors.append(mean_squared_error(y_val, y_pred))

            median_error = np.median(fold_errors)
            if median_error < best_error:
                best_error = median_error
                best_gamma = gamma

        return best_gamma