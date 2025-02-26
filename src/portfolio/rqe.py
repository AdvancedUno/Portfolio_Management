import numpy as np
from scipy.optimize import minimize
from scipy.optimize import differential_evolution

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


def rqe(weights, cov_matrix):
    """
    Calculate the RQE value

    Param:
        weights : Portfolio weights
        cov_matrix : Covariance matrix

    Returns:
        RQE value
    """
    portfolio_var = weights @ cov_matrix @ weights
    sum_weight_var = np.sum(weights * np.diag(cov_matrix))
    return portfolio_var - sum_weight_var


def optimize_rqe(cov_matrix, initial_weights):
    """
    Optimizes RQE portfolio
    
    Parameters:
        The covariance matrix of asset returns.

    Returns:
       Optimized RQE weights
    """
    constraints = (
        {"type": "eq", "fun": lambda x: np.sum(x) - 1},
        {"type": "ineq", "fun": lambda x: x}  
    )

    def obj(x):
        # Maximize RQE is basically minimize negative of RQE
        return rqe(x, cov_matrix)

    result = minimize(obj, initial_weights, method="SLSQP", constraints=constraints)
    return result.x



def optimize_rqe_global(cov_matrix, n_restarts=300):
    """
    Optimizes RQE portfolio using a multi-start b/c local min
    
    Parameters:
        The covariance matrix of asset returns.

    Returns:
       Optimized RQE weights
    """
        
    best_weights = None
    best_obj = float('inf')  # We minimize the negative RQE to maximize RQE
    
    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        #{'type': 'ineq', 'fun': lambda x: x}
    )
    
    for _ in range(n_restarts):
        x0 = np.random.dirichlet(np.ones(cov_matrix.shape[1]))
        result = minimize(
            rqe,  # Minimize negative RQE to maximize it
            x0,
            method='SLSQP',
            args=(cov_matrix), 
            constraints=constraints,
            bounds=[(0, 1) for _ in range(cov_matrix.shape[1])]
        )
        if result.fun < best_obj:
            best_obj = result.fun
            best_weights = result.x
    
    return best_weights



equal_weights = np.ones(test_matrix1.shape[1]) / test_matrix1.shape[1] 

optimized_enb_weights = optimize_rqe_global(test_matrix1)
print("Optimized RQE Weights Test 1 :", np.round(optimized_enb_weights,3))


optimized_enb_weights = optimize_rqe_global(test_matrix2)
print("Optimized RQE Weights Test 2 :", np.round(optimized_enb_weights,3))

code_val = rqe(optimized_enb_weights, test_matrix2)
expected_w = [0.248, 0.346, 0, 0.405]
expected_val = rqe(expected_w, test_matrix2)
print("Expected Value: ", expected_val)
print("Code Value: ", code_val)

optimized_enb_weights = optimize_rqe_global(test_matrix3)
print("Optimized RQE Weights Test 3 :", np.round(optimized_enb_weights,3))

