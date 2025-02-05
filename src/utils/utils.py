import numpy as np
import pandas as pd

# Volatilities of assets
volatilities = np.array([0.1, 0.2, 0.3, 0.4])

# Correlation matrix
correlation_matrix = np.array([
    [1.0,  -0.8,  0.0,  0.0],  
    [-0.8,  1.0,  0.0,  0.0],  
    [0.0,  0.0,  1.0,  0.5],  
    [0.0,  0.0,  0.5,  1.0],  
])

covariance_matrix = np.outer(volatilities, volatilities) * correlation_matrix

cov_matrix_df = pd.DataFrame(covariance_matrix, columns=["Asset 1", "Asset 2", "Asset 3", "Asset 4"], 
                             index=["Asset 1", "Asset 2", "Asset 3", "Asset 4"])

print(cov_matrix_df)