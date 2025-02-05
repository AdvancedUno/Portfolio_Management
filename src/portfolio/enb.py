import numpy as np
from scipy.optimize import minimize



# Test cases for the paper
# For some reason the testMatrix3 is returning different output than expacted.
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



def effective_number_of_bets(weights, cov_matrix):

    """
    Calculate the Effective Number of Bets (ENB) of portfolio.

    Param:
        weights : Portfolio weights
        cov_matrix : Covariance matrix

    Returns:
        ENB value
    """   


    # Diagonal matrix of standard deviations
    D = np.diag(np.sqrt(np.diag(cov_matrix)))
    invD = np.diag(1 / np.diag(D))  

    # Eigen dcomposition of the covariance matrix
    eigenvalues, P = np.linalg.eig(cov_matrix)
    Lambda = np.diag(np.sqrt(eigenvalues))
    LambdaInv = np.diag(1 / np.sqrt(eigenvalues))

    # Singular Value Decomposition
    Transicion = Lambda @ P.T @ D
    U, S_diag, Vt = np.linalg.svd(Transicion)
    S = np.diag(S_diag)

    # Calculate A and A^(-1)
    A = P @ LambdaInv @ U @ Vt @ D
    invA = invD @ Vt.T @ U.T @ Lambda @ P.T

    # Calculate A^(-1) * weights
    invAw = invA @ weights

    # Calculate portfolio variance
    wSIGMAw = weights.T @ cov_matrix @ weights

    # Calculate pk for ENB calculation
    pk = np.diag(cov_matrix) * (invAw ** 2) / wSIGMAw

    # Calculate ENB
    lnpk = np.log(pk)
    pklnpk = pk @ lnpk
    resultENB = np.exp(-pklnpk)

    return resultENB



def optimize_enb(cov_matrix, initial_weights):

    """
    Optimize the Effective Number of Bets (ENB).

    Param:
        cov_matrix : Covariance matrix
        initial_weights : Initial weights

    Returns:
        Optimized weights.
    """

    # Constraints are sum of weights need to be 1 and all the weights >= 0
    constraints = (
        {"type": "eq", "fun": lambda x: np.sum(x) - 1}, 
        {"type": "ineq", "fun": lambda x: x}  
    )

    # Objective function is to maximize ENB wich is minimizing ENB
    def obj(x):
        return -effective_number_of_bets(x, cov_matrix)

    # Optimize
    # Using SLSQP make senes for me
    result = minimize(obj, initial_weights, method="SLSQP", constraints=constraints)

    return result.x



optimized_enb_weights = optimize_enb(test_matrix1, equal_weights)
print("Optimized ENB Weights Test 1 :", np.round(optimized_enb_weights,3))


optimized_enb_weights = optimize_enb(test_matrix2, equal_weights)
print("Optimized ENB Weights Test 2 :", np.round(optimized_enb_weights,3))


optimized_enb_weights = optimize_enb(test_matrix3, equal_weights)
print("Optimized ENB Weights Test 3 :", np.round(optimized_enb_weights,3))


