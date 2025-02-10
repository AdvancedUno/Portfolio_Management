import numpy as np
from scipy.optimize import minimize

# Example covariance matrices for testing

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

testMat = test_matrix1

# Equal weights in the beginning
equal_weights = np.ones(testMat.shape[1]) / testMat.shape[1]


def erc(weights, cov_matrix):
    """
    Calculate the objective function for Equal Risk Contribution (ERC).

    Param:
        weights : Portfolio weights
        cov_matrix : Covariance matrix

    Returns:
        Objective function value (sum of squared differences in risk contributions).
    """
    # Portfolio variance
    portfolio_variance = weights.T @ cov_matrix @ weights

    # Marginal risk contributions
    marginal_risk = cov_matrix @ weights

    # Risk contributions
    risk_contributions = weights * marginal_risk / np.sqrt(portfolio_variance)

    # Objective: minimize the sum of squared differences in risk contributions
    target_risk_contribution = 1 / len(weights)  # Equal risk contribution target
    obj_value = np.sum((risk_contributions - target_risk_contribution) ** 2)

    return obj_value

def optimize_erc(cov_matrix, initial_weights):
    """
    Optimize the Equal Risk Contribution (ERC) portfolio.

    Param:
        cov_matrix : Covariance matrix
        initial_weights : Initial weights

    Returns:
        Optimized weights.
    """
    # Constraints are the sum of weights add up to 1 and all the weights >= 0
    constraints = (
        {"type": "eq", "fun": lambda x: np.sum(x) - 1},
        {"type": "ineq", "fun": lambda x: x} 
    )
    
    # Objective function is to minimize erc value
    def obj(x):
        return erc(x, cov_matrix)

    # Optimize using SLsSQP like enb
    result = minimize(obj, initial_weights, method="SLSQP", constraints=constraints)

    return result.x



optimized_erc_weights = optimize_erc(test_matrix1, equal_weights)
print("Optimized ERC Weights Test 1 :", np.round(optimized_erc_weights,3))

optimized_erc_weights = optimize_erc(test_matrix2, equal_weights)
print("Optimized ERC Weights Test 2 :", np.round(optimized_erc_weights,3))

optimized_erc_weights = optimize_erc(test_matrix3, equal_weights)
print("Optimized ERC Weights Test 3 :", np.round(optimized_erc_weights,3))