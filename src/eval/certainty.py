import numpy as np

def certainty_equivalent(returns, gamma, risk_free_rate=0.02):
    """
    Compute the Certainty-Equivalent Return.
    
    :param returns: Array of portfolio excess returns.
    :param gamma: Risk aversion coefficient.
    :param risk_free_rate: Risk-free rate.
    :return: Certainty value.
    """
    
    mu_h = np.mean(returns) - risk_free_rate
    sigma_h2 = np.var(returns, ddof=1)  # Use sample variance

    return mu_h - (gamma / 2) * sigma_h2



simulated_returns = np.random.normal(0.08, 0.15, 100)  # 100 periods of random returns

gamma_value = 3

CEQ_value = certainty_equivalent(simulated_returns, gamma_value)

print(f"Certainty-Equivalent Return: {CEQ_value:.4f}")
