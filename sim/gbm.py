import numpy as np
import hashlib
from typing import Optional
import yfinance as yf

def rng_from_string(seed):
    if not isinstance(seed, str):
        seed = str(seed)
    seed = int(hashlib.sha256(seed.encode("utf-8")).hexdigest(), 16)
    seed = seed % (2**32)
    return np.random.default_rng(seed)

def simulate_gbm(S_0 : float, volatility : float, drift : float, n_rounds : int, seed:  Optional[str] = None): 

    #S_0        : Initial Asset Price - Rules S_0 > 0
    #volatility : Volatility Parameter - σ (sigma)
    #drift      : Drift Parameter - Per Step
    #n_rounds   : Number of time steps to simulate
    #seed       : Optional random seed to make paths reproducible

    if S_0 <= 0 or not np.isfinite(S_0): 
        raise ValueError(f"Error: S_0 cannot be <= 0, must be numeric and finite (Current Value S_0 = {S_0})") 
    if volatility < 0 or not np.isfinite(volatility): 
        raise ValueError(f"Error: volatility cannot be < 0, must be numeric and finite (Current Value volatility = {volatility})") 
    if not np.isfinite(drift): 
        raise ValueError(f"Error: Drift must be numeric and finite (Current Value drift = {drift})") 
    if n_rounds <= 0: 
        raise ValueError("n_rounds must be a positive integer.") 
    if seed is None:
        seed = str(np.random.SeedSequence().entropy)


    rng = rng_from_string(seed) #We want reproducible GBM paths so using a seed allows reproducibility, conceptually Gaussian shocks (epsilon)
    increments = (drift - 0.5 * volatility**2) + volatility * rng.standard_normal(n_rounds) #Generates GBM increments Timestep of Zt = (drift - 0.5 * volatility ** 2) + volatility * epsilon
    
    logS = np.empty(n_rounds + 1) #empty array that will be used to store the initial log price and shock path - Z_t = log(S_t)
    logS[0] = np.log(S_0) #Z at time Z = log S0
    logS[1:] = logS[0] + np.cumsum(increments)

    S_t = np.exp(logS) #Builds price path, ensures prices are strictly positive and distribution is lognormal
    return S_t  
#-----------------------------

#------ Historical Data ------
def simulate_from_yfinance(ticker: str, start_date: str, n_rounds: int, interval: str, price_col: str = "Close", auto_adjust: bool = True):

    #ticker     : The ticker to the stock they want to simulate
    #start_date : starting date
    #n_rounds   : number of rounds
    #interval   : interval
    #price_col  : pricing column
    #auto_adjust: auto_adjust

    hist = yf.Ticker(ticker).history(start=start_date, interval=interval, auto_adjust=auto_adjust) #Downloads pandas dataframe from yfinance indexed by time - Format : Datetime | Open | High | Low | Close | Volume

    if hist.empty: #Checks if histogram is empty (e.g. invalid ticker, invalid interval)
        raise ValueError(f"Data not found from referenced ticker '{ticker}' (interval={interval}).") #Raises a value error displaying that the data from the referenced ticker was not found
    if price_col not in hist.columns: #Checks if the price column you are referencing exists
        raise ValueError(f"Column Type '{price_col}' not found within history columns: {list(hist.columns)}") #Raises a value error displaying that the referenced column type doesn't exist
    
    prices = hist[price_col].dropna().to_numpy(dtype=float) #Select chosen column, drop missing values, convert to numpy array

    if prices.size < (n_rounds + 1): #Ensure there is enough data for the number of rounds requested
        raise ValueError(f"Not enough data for '{ticker}'. Need {n_rounds+1} points, got {prices.size}") #Raises value error
    
    S_t = prices[:(n_rounds + 1)].copy() #Builds the historical path
    Z = np.zeros(n_rounds + 1, dtype = float) #Zeros array that stores the initial log price and shock path - Z_t = log(S_t)
    Z[1:] = np.diff(np.log(S_t)) # Builds and stores the historical log returns path
    return S_t 
#---------------------------

