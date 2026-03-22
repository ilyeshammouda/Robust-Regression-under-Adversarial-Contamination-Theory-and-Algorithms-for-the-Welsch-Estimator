import numpy as np
from scipy.optimize import minimize


class OLS:
    """
    Estimates beta using the Ordinary Least Squares (L2) loss function.

    Supports two approaches:
        - optimizer_approach: solves the optimization problem directly via a scipy solver.
        - gradient_descent_approach: optimizes via gradient descent.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, sigma: float = 1.0):
        self.X = X
        self.y = y
        self.sigma = sigma
        self.n, self.p = X.shape

    # ------------------------------------------------------------------ #
    # Loss and gradient
    # ------------------------------------------------------------------ #

    def l2_loss(self, beta: np.ndarray) -> float:
        """Compute the mean squared error (L2 loss)."""
        return np.linalg.norm(self.y - self.X @ beta) ** 2 / self.n

    def l2_loss_gradient(self, beta: np.ndarray) -> np.ndarray:
        """Compute the gradient of the L2 loss."""
        residuals = self.y - self.X @ beta
        return (-2 / self.n) * (self.X.T @ residuals)

    # ------------------------------------------------------------------ #
    # Optimizer approach
    # ------------------------------------------------------------------ #

    def optimizer_approach(
        self,
        initial_guess: np.ndarray = None,
        method: str = "L-BFGS-B",
        max_iter: int = 1000,
    ) -> np.ndarray:
        """
        Minimize the L2 loss using a scipy solver.

        Returns:
            beta_hat: estimated coefficients.
        """
        if initial_guess is None:
            initial_guess = np.zeros(self.p)

        result = minimize(
            self.l2_loss,
            initial_guess,
            method=method,
            options={"maxiter": max_iter},
        )
        return result.x

    # ------------------------------------------------------------------ #
    # Gradient descent approach
    # ------------------------------------------------------------------ #

    def gradient_descent_approach(
        self,
        initial_guess: np.ndarray = None,
        beta_star: np.ndarray = None,
        learning_rate: float = 0.001,
        iterations: int = 1000,
    ) -> tuple:
        """
        Minimize the L2 loss using gradient descent.

        Args:
            initial_guess: starting point for optimization.
            beta_star: true coefficients (for tracking convergence on simulated data).
            learning_rate: step size.
            iterations: number of gradient descent steps.

        Returns:
            If beta_star is provided: (beta_hat, convergence_distances).
            Otherwise: beta_hat.
        """
        if initial_guess is None:
            initial_guess = np.random.normal(
                loc=0, scale=np.sqrt(2 / (self.p + self.n)), size=self.p
            )

        beta = initial_guess.copy()
        convergence = [np.linalg.norm(beta - beta_star)] if beta_star is not None else None

        for _ in range(iterations):
            gradient = self.l2_loss_gradient(beta)
            beta -= learning_rate * gradient

            if convergence is not None:
                convergence.append(np.linalg.norm(beta - beta_star))

        if convergence is not None:
            return beta, convergence
        return beta