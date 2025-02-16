import numpy as np


def turnover(weights_matrix):
    """
    Compute portfolio turnover.

    :param weights_matrix: 2D array of portfolio weights
    :return: Turnover value
    """

    T, N = weights_matrix.shape  # T is time periods, N is assets
    turnover_sum = np.sum(np.abs(np.diff(weights_matrix, axis=0)))

    return turnover_sum / (T - 1)  # Normalize over time periods


simulated_returns = np.random.normal(0.08, 0.15, 100)  

simulated_weights = np.random.dirichlet(np.ones(4), size=100)


turnover_value = turnover(simulated_weights)

print(f"Turnover: {turnover_value:.4f}")
