import numpy as np
from scipy.optimize import minimize


test_matrix1 = np.array([
    [0.01, 0.016, 0, 0],
    [0.016, 0.04, 0, 0],
    [0, 0, 0.09, -0.06],
    [0, 0, -0.06, 0.16]
])

test_matrix2 = np.array([
    [0.01, -0.016, 0, 0],
    [-0.016, 0.04, 0, 0],
    [0, 0, 0.09, 0.06],
    [0, 0, 0.06, 0.16]
])

test_matrix3 = np.array([
    [0.01, 0.016, 0, 0],
    [0.016, 0.04, 0, 0],
    [0, 0, 0.09, -0.108],
    [0, 0, -0.108, 0.16]
])

initial_weights = np.ones(test_matrix1.shape[1]) / test_matrix1.shape[1]

def calculate_portfolio_variance(weights, cov_matrix):
    """
    Calculate the portfolio variance given weights and covariance matrix.
    
    Param:
        weights : Portfolio weights
        cov_matrix : Covariance matrix

    Returns:
        Dot product of weights and covariance matrix
    """
    return weights.T @ cov_matrix @ weights

def optimize_min_variance(cov_matrix, initial_weights):
    """
    Optimize to find the minimum variance portfolio with weights sum up to 1.

    Param:
        initial_weights : initial weights
        cov_matrix : Covariance matrix

    Returns:
        Optimized weights.
    """
    # Constraint: weights sum to 1
    constraints = (
        {"type": "eq", "fun": lambda x: np.sum(x) - 1},
        {"type": "ineq", "fun": lambda x: x} 
    )

    def obj(x):
        return calculate_portfolio_variance(x, cov_matrix)

    result = minimize(
        obj,
        initial_weights,
        method='SLSQP',
        constraints=constraints
    )

    return result.x


optimized_mean_var_weights = optimize_min_variance(test_matrix1, initial_weights)
print("Optimized Mean Var Weights Test 1 :", np.round(optimized_mean_var_weights,3))

optimized_mean_var_weights = optimize_min_variance(test_matrix2, initial_weights)
print("Optimized Mean Var Weights Test 2 :", np.round(optimized_mean_var_weights,3))

optimized_mean_var_weights = optimize_min_variance(test_matrix3, initial_weights)
print("Optimized Mean Var Weights Test 3 :", np.round(optimized_mean_var_weights,3))

