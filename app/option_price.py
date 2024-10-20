import numpy as np
import yfinance as yf
from scipy.stats import norm
from arch import arch_model

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    return call_price

def garch_estimate_volatility(price_data):
    returns = 100 * price_data.pct_change().dropna()
    model = arch_model(returns, vol='Garch', p=1, q=1)
    model_fit = model.fit(disp="off")
    forecast = model_fit.forecast(horizon=1)
    volatility = np.sqrt(forecast.variance.values[-1, :][0])
    
    return volatility / 100

