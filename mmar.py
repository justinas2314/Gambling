# import
import nolds
import pandas as pd
import numpy as np
import math
from fbm import *
from fbm import FBM
from tqdm import tqdm

def segment_data(data, num_segments):
    """
    Splits the data into multiple segments.
    """
    len_segment = len(data) // num_segments
    return [data[i:i+len_segment] for i in range(0, len(data), len_segment) if len(data[i:i+len_segment]) == len_segment]

def calculate_hurst_for_segments(data, num_segments):
    """
    Calculate the Hurst exponent for the segment
    """
    segments = segment_data(data, num_segments)
    hurst_values = [nolds.hurst_rs(seg) for seg in segments]
    return hurst_values

def define_time_window(min_window:int, max_window:int, base:float=10, interval:float=0.25):
    '''
    @param: min_window: the minimum required period window
    @param: max_window: the maximum required period window
    @param: base: the base value to be powered of to generate the series
    @param: interval: the interval used to seperate between the min window and max window

    @return: window_sizes: the array that contains the transformed trade time interval
    '''
    window_sizes = list(map(
        lambda x: int(base**x),
        np.arange(math.log10(min_window), math.log10(max_window), interval)))
    
    return window_sizes

def calculate_scaling_exponent(delta, x_t, q):
    '''
    @param: delta array: time interval range
    @param: x_t array: the time series
    @param: q float: the power q
    
    @return: Fq pd.DataFrame: the partition function values for different delta and q
    @return: tau_q_list list: the scaling exponent
    '''
    # Initialize a 2D array sigma to store the partition function values.
    # row for different delta
    Fq = [[0 for x in range(len(delta))] for y in range(len(q))]
    
    # Loop through each exponent q[k] and time lag delta[j].
    # For each q and delta, compute the partition function by:
    # i) Iterating through the time series in chunks of size delta[j]
    # ii) Calculating the absolute difference between each point and the point delta[j] steps ahead
    # iii) Summing the q[k] power of these differences
    for k in range (0, len(q)):
        # if k%30==0: # dividable by 30
        #     print("calculating q=" + str(k) + ' out of ' + str(len(q)-1))
            
        for j in range (0,len(delta)):
            
            for i in range (0,len(x_t)-1):
                if i < int((len(x_t)-1)/delta[j]):
                    Fq[k][j]=Fq[k][j] + abs(x_t[i*delta[j]+delta[j]]-x_t[i*delta[j]])**q[k]

    Fq=pd.DataFrame(Fq)

    for i in range(0,len(q)):
        Fq.rename(index={Fq.index[i]:q[i]}, inplace=True)
    for i in range(len(delta)-1,-1,-1):
        Fq.rename(columns={Fq.columns[i]:delta[i]}, inplace=True)

    # print("Finished calculating the partition values Fq")

    # Power-law regression on log Fq(q, delta) vs log delta to get scaling exponent τ(q)
    # If the signal has scaling properties, Fq will follow a power law:
    # Fq ~ delta^τ(q)
    tau_q_list = []
    for i,row in Fq.iterrows():
        Fq_matrix = np.vstack([np.log10(row.values), np.ones(len(row))]).T
        tau_q, c = np.linalg.lstsq(Fq_matrix, np.log10(delta), rcond=-1)[0]
        tau_q_list.append(tau_q)

    return Fq, tau_q_list


def estimate_multifractal_spectrum(tau_q_list:list, q:list, start_of_list:int, end_of_list:int):
    '''
    @param: tau_q_list: the list that containing the scaling exponents for the returns
    @param: q: the statistical moments
    @param: start_of_q_list: the start index of the list for comparison
    
    @return: F_A pd.DataFrame: the multifractal spectrum values for different q
    @return: parameters_of_spectrum tuple: contains width_of_spectrum, holder_exponent, asymmetry_of_spectrum which are the parameters value of the spectrum
    '''
    tau_q_estimated = np.polyfit(q[start_of_list:end_of_list], tau_q_list[start_of_list:end_of_list], 2)

    F_A = [0 for x in range(len(q)-10)]
    p = [0 for x in range(len(q)-10)]

    a = tau_q_estimated[0]
    b = tau_q_estimated[1]
    c = tau_q_estimated[2]

    for i in range(0, len(q)-10):
        p[i] = 2*a*q[i]+b
        F_A[i] = ((p[i]-b)/(2*a))*p[i] - (a*((p[i]-b)/(2*a))**2 + b*((p[i]-b)/(2*a)) + c)

    F_A = pd.DataFrame(F_A)
    F_A.rename(columns={F_A.columns[0]:"f(a)"}, inplace=True)
    F_A['p'] = p

    # print("Using the range of q's from " + str(q[start_of_list]) + " to " + str(q[end_of_list]) + ":")
    # tau_q_estimated are the coefficients (a,b,c) from fitting the quadratic model to τ(q).
    # print("The estimated parameters for tau(q) are: \n" + str(tau_q_estimated))
    
    # the three estimated parameters for f(a) are derived from the Legendre transform relations:
    # i) 1/(4*a) is the width of the f(α) spectrum
    width_of_spectrum = 1/(4*a)
    # ii) (-2b)/(4a) is the position of the peak α0 == most probable Hölder exponent
    holder_exponent = (-2*b)/(4*a)
    # iii) (-4ac+b^2)/(4*a) is related to asymmetry in f(α)
    asymmetry_of_spectrum = (-4*a*c+b**2)/(4*a)
    # they give key properties of f(α) spectrum
    # print("\nThus, the estimated parameters for f(a) are: \n width_of_spectrum: " + str(width_of_spectrum) + ", \n holder_exponent: "  + str(holder_exponent) + ", \n asymmetry_of_spectrum: "+ str(asymmetry_of_spectrum))

    return F_A, (width_of_spectrum, holder_exponent, asymmetry_of_spectrum)

def calculate_lognormal_cascade(layers:int, v:float, ln_lambda:float, ln_sigma:float):
    '''
    @param: layers: the layers of the branching of lognormal cascade
    @param: v: the value for branching at each step
    @param: ln_lambda: the mean of the log-normal distribution for drawing the random lognormal weight
    @param: ln_sigma: the standard derivation of the log-normal distribution for drawing the random lognormal weight
    
    @return: v array: the values of the resultant lognormal cascade
    '''
    layers = layers - 1
    
    m0 = np.random.lognormal(ln_lambda,ln_sigma)
    m1 = np.random.lognormal(ln_lambda,ln_sigma)
    m0 = m0/(m0+m1)
    m1 = m1/(m0+m1)
    
    M = [m0, m1]

    if (layers >= 0):
        d=[0 for x in range(0,2)]
        for i in range(0,2):
            d[i] = calculate_lognormal_cascade(layers, (M[i]*v), ln_lambda, ln_sigma)

        v = d

    return v

def calculate_trading_time(layers:int, lognormal_cascade:list):
    '''
    @param: layers: the layers of the branching of lognormal cascade
    @param: lognormal_cascade: list of the lognormal cascade value
    
    @return: trading_time: list of the trading time calculated which is delta_t
    '''
    trading_time = 2**layers*np.cumsum(lognormal_cascade)/sum(lognormal_cascade)
    return trading_time

def calculate_magnitude_parameter(initial_value:float, eps:float, steps:float, number_of_path:int, real_std:float, layers:int, hurst_exponent:float):
    '''
    @param: initial_value: the initial value of the magnitude parameter
    @param: eps: the threshold of the difference in order for the magnitude parameter to be accepted
    @param: steps: the portion of the difference to be adjusted from the magnitude parameter
    @param: number_of_path: the number of paths for the simulation
    @param: real_std: the std of the underlying to be compared with
    @param: layers: the layers of the branching of lognormal cascade
    @param: hurst_exponent: the hurst exponent of the underlying
    
    @return: magnitude_parameter: the resultant magnitude parameter
    '''
    diff = np.inf
    magnitude_parameter = initial_value

    while abs(diff) > eps:
        std_list = []
        for nb in range(number_of_path): # excluding tqdm for a less verbose output
            new_fbm_class = FBM(n = 10*2**layers+1, hurst = hurst_exponent, length = magnitude_parameter, method='daviesharte')
            new_fbm_simulation = new_fbm_class.fbm()
            std_list.append(np.std(new_fbm_simulation))
        diff = real_std - np.median(std_list)
        # print('Diff: ', diff)
        if abs(diff) > eps:
            magnitude_parameter += diff * steps
            # print('new magnitude_parameter:', magnitude_parameter)

    return  magnitude_parameter

def calculate_mmar_returns(S0:float, number_of_path:int, layers:int, hurst_exponent:float, trading_time:list, magnitude_parameter:float, time_window_base:float=10):
    '''
    @param: S0: the initial value of the underlying asset's price
    @param: number_of_path: the number of paths for the simulation
    @param: layers: the layers of the branching of lognormal cascade
    @param: hurst_exponent: the hurst exponent of the underlying
    @param: trading_time: the trading time
    @param: magnitude_parameter: the magnitude parameter to control the length of the FBM process
    @param: time_window_base: the base for defining the time window
    
    @return: mmar_returns: the simulated return of mmar
    @return: mmar_prices: the simulated prices of the underlying assets
    '''
    mmar_returns = []
    mmar_prices = []

    for nb in tqdm(range(number_of_path)):
        new_fbm_class = FBM(n = 10*2**layers+1, hurst = hurst_exponent, length = magnitude_parameter, method='daviesharte')
        new_fbm_simulation = new_fbm_class.fbm()
        new_fbm_simulation = new_fbm_simulation[1:]
        
        # --- MMAR returns's ---
        simulated_xt_array = [0 for x in range(0, len(trading_time))]
        for i in range(0, len(trading_time)):
            simulated_xt_array[i] = new_fbm_simulation[int(trading_time[i]*10)]
        mmar_returns.append(simulated_xt_array)
            
        # --- PRICES ---
        simulated_prices_array = S0 * np.exp(simulated_xt_array)
        mmar_prices.append(simulated_prices_array)

    return mmar_returns, mmar_prices

def option_pricer(paths, strike, r, T, option_type='call'):
    """
    Calculate the option price (european) given simulated paths.
    
    Parameters:
    - paths: An array of simulated asset paths. 
    - strike: Strike price of the option.
    - r: Risk-free rate.
    - T: Time to maturity in years.
    - option_type: call or put.
    
    Returns:
    - Option price
    """
    
    if isinstance(paths, list):
        paths = np.array(paths)
        
    # Get the asset prices at maturity (last column of paths matrix)
    S_T = paths[:, -1]
    
    # Calculate the payoff for each path
    if option_type == 'call':
        payoffs = np.maximum(S_T - strike, 0)
    elif option_type == 'put':
        payoffs = np.maximum(strike - S_T, 0)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")
    
    # Average the payoffs and discount back to today
    option_price = np.exp(-r * T) * np.mean(payoffs)
    
    return option_price
