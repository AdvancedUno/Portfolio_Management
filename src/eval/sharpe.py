import numpy as np

def sharpe_ratio(returns, risk_free_rate=0.02):
    """
    Compute Sharpe Ratio.
    
    :param returns: Array of portfolio excess returns
    :param risk_free_rate: Risk-free rate
    :return: Sharpe Ratio
    """
    mu_h = np.mean(returns) - risk_free_rate
    sigma_h = np.std(returns, ddof=1)  # Use sample standard deviation

    return mu_h / sigma_h

simulated_returns = np.random.normal(0.08, 0.15, 100)  

SR_value = sharpe_ratio(simulated_returns)
print(f"Sharpe Ratio: {SR_value:.4f}")
