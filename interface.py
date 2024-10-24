import math
import warnings

import yfinance
import numpy as np
import matplotlib.pyplot as plt
from sklearn.exceptions import UndefinedMetricWarning

import mmar
from mmar import option_pricer  # cia as cookinu

# sicia irgi goated linai bet siti pastealinti
warnings.simplefilter(action='ignore', category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", message=".*RANSAC did not reach consensus.*", category=RuntimeWarning)


def plot_mmar_paths(paths, title, xlabel, ylabel):
    plt.figure(figsize=(10, 6))

    for path in paths:
        plt.plot(path)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()


def get_df_yf(ticker, start_date, end_date):
    return yfinance.download(ticker, start_date, end_date)


def get_hurst(series):
    return sum(lst := mmar.calculate_hurst_for_segments(series, 625)) / len(lst)


# step 1
def get_close_return(prices):
    output = prices.copy()
    output.loc[output < 0] = 0.0001
    return np.log(output / output.shift()).dropna()


# step 4
def get_tau_q_list(prices, q):
    window_sizes = mmar.define_time_window(min_window=10, max_window=len(prices), base=10, interval=0.25)
    # fq, tau_q_list = mmar.calculate_scaling_exponent(window_sizes, prices, q)
    return mmar.calculate_scaling_exponent(window_sizes, prices, q)[1]


# step 7
def get_trading_time(prices, h_exp, q, days_for_simulation):
    tau_q_list = get_tau_q_list(prices, q)
    parameters_of_spectrum = mmar.estimate_multifractal_spectrum(tau_q_list, q, 0, len(q) - 1)[1]
    sim_l = parameters_of_spectrum[1] / h_exp

    # σ2 = 2(λ — 1) / ln[b]
    simulated_sigma = np.sqrt(2 * (sim_l - 1) / np.log(2))
    # find the k when b == 2
    k = math.ceil(np.log2(days_for_simulation))  # k value
    new_cascade = mmar.calculate_lognormal_cascade(layers=k, v=1, ln_lambda=np.log(sim_l),
                                                   ln_sigma=np.log(simulated_sigma))
    # tingiu tikrinti ar sita pakeitus kazkas blogo neatsitiks
    # bet cia durniausiai atrodantis line of code kuri teko matyti siame mano gyvenimo desimtmetyje
    new_cascade = list(np.array(new_cascade).flat)
    return mmar.calculate_trading_time(layers=k, lognormal_cascade=new_cascade)[:days_for_simulation]


# step 13
def simulate_fbm(close_return, h_exp, layers):
    return mmar.calculate_magnitude_parameter(initial_value=0.5, eps=0.01, steps=0.5, number_of_path=100,
                                              real_std=np.std(close_return), layers=layers, hurst_exponent=h_exp)


# step 14
def simulate_mmar(s0, prices, days_for_simulation, num_paths, graph=False):
    q = np.linspace(0.01, 3, 120)
    h_exp = get_hurst(prices)
    close_return = get_close_return(prices)
    layers = math.ceil(np.log2(days_for_simulation))
    trading_time = get_trading_time(prices, h_exp, q, days_for_simulation)
    magnitude_parameter = simulate_fbm(close_return, h_exp, layers)
    mmar_returns, prices_paths = mmar.calculate_mmar_returns(S0=s0, number_of_path=num_paths, layers=layers,
                                                             hurst_exponent=h_exp, trading_time=trading_time,
                                                             magnitude_parameter=magnitude_parameter)
    if graph:
        plot_mmar_paths(mmar_returns, 'Simulated MMAR Returns', 'Days', 'Returns')
        plot_mmar_paths(prices_paths, 'Simulated MMAR Prices', 'Days', 'Price')
        plot_mmar_paths([s0 * np.exp(np.mean(mmar_returns, axis=0))],
                        'Prices derived from the mean return', 'Time\n(days)', 'Prices')
    return mmar_returns, prices_paths


def price_options_for_strikes(paths, center, step=5, num_strikes=5, r=0.05, t=1):
    option_prices = {}
    # Calculate prices for many strikes
    for i in range(1, num_strikes + 1):
        strike = center - i * step
        option_prices[strike] = mmar.option_pricer(paths, strike, r, t, option_type='put')

    # Calculate prices for strikes at and above the center
    for i in range(num_strikes + 1):
        strike = center + i * step
        option_prices[strike] = mmar.option_pricer(paths, strike, r, t, option_type='call')

    return option_prices
