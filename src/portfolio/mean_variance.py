import numpy as np
from scipy.optimize import minimize

def calculate_portfolio_variance(weights, cov_matrix):
    """Calculate the portfolio variance given weights and covariance matrix."""
    return np.dot(weights.T, np.dot(cov_matrix, weights))

def optimize_min_variance(cov_matrix, initial_weights):
    """Optimize to find the minimum variance portfolio with weights summing to 1."""
    # Constraint: weights sum to 1
    constraints = (
        {"type": "eq", "fun": lambda x: np.sum(x) - 1},
        {"type": "ineq", "fun": lambda x: x} 
    )
    # Bounds: allow short selling by default (no bounds)
    bounds = tuple((None, None) for _ in range(len(initial_weights)))
    
    result = minimize(
        calculate_portfolio_variance,
        initial_weights,
        args=(cov_matrix,),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    return result

# Example inputs
test_matrix1 = np.array([
    [0.01, 0.016, 0, 0],
    [0.016, 0.04, 0, 0],
    [0, 0, 0.09, -0.06],
    [0, 0, -0.06, 0.16]
])

initial_weights = np.ones(test_matrix1.shape[1]) / test_matrix1.shape[1]

# Calculate initial portfolio variance
initial_variance = calculate_portfolio_variance(initial_weights, test_matrix1)
print("Initial Portfolio Variance:", initial_variance)

# Optimize portfolio
result = optimize_min_variance(test_matrix1, initial_weights)


optimal_weights = result.x
min_variance = result.fun
print("Optimal Weights:", optimal_weights)

