import numpy as np

def sharpe_ratio(returns, risk_free_rate=0.02):
    """
    Compute Sharpe Ratio.

    Param:
        returns : Array of portfolio returns
        risk_free_rate : Risk-free rate

    Returns:
        Sharpe Ratio value
    """
    mu = np.mean(returns) - risk_free_rate
    sigma = np.std(returns, ddof=1) 

    return mu / sigma

test_returns = np.random.normal(0.08, 0.15, 100)  

sharpe_ratio_value = sharpe_ratio(test_returns)
print(f"Sharpe Ratio: ",  np.round(sharpe_ratio_value,3))
