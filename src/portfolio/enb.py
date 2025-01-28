import numpy as np
from scipy.optimize import minimize



# Test cases for the paper
# For some reason the testMatrix3 is returning different output than expacted.
testMatrix1 = np.array([
    [0.01, -0.016, 0, 0],
    [-0.016, 0.04, 0, 0],
    [0, 0, 0.09, 0.06],
    [0, 0, 0.06, 0.16]
])

testMatrix2 = np.array([
    [0.01, 0.016, 0, 0],
    [0.016, 0.04, 0, 0],
    [0, 0, 0.09, -0.06],
    [0, 0, -0.06, 0.16]
])


testMatrix3 = np.array([
    [0.01, -0.016, 0, 0],
    [-0.016, 0.04, 0, 0],
    [0, 0, 0.09, -0.06],
    [0, 0, -0.06, 0.16]
])


# Equal weights in the beginning
equal_weights = np.ones(testMatrix1.shape[1]) / testMatrix1.shape[1] 

print(equal_weights)

def effective_number_of_bets(weights, cov_matrix):


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

    # Constraints are sum of weights need to be 1, weights >= 0
    constraints = (
        {"type": "eq", "fun": lambda x: np.sum(x) - 1},  # Sum of weights need to be 1
        {"type": "ineq", "fun": lambda x: x}  # Weights >= 0
    )

    # Objective function is to maximize ENB wich is minimizing ENB
    def obj(x):
        return -effective_number_of_bets(x, cov_matrix)

    # Optimize
    # Using SLSQP make senes for me
    result = minimize(obj, initial_weights, method="SLSQP", constraints=constraints)

    return result.x



optimized_enb_weights = optimize_enb(testMatrix1, equal_weights)
print("Optimized ENB Weights:", optimized_enb_weights)




