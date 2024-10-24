import matplotlib.pyplot as plt

import interface

# prices turi buti pandas series[numpy.float64] tipo daiktas
# (po to gal pakeisiu jeigu noresiu aiskintis ka finansininkai cookino ir kodel kodas toks letas)
prices = interface.get_df_yf('CL=F', '1990-01-01', '2023-08-24')['Adj Close']
print(type(prices))
prices.plot()
plt.show()
last_price = prices.iloc[-1]
mmar_returns, prices_paths = interface.simulate_mmar(last_price, prices, 252, 1000, graph=True)
option_prices = interface.price_options_for_strikes(prices_paths, round(last_price))

print("Call option price: ", interface.option_pricer(prices_paths, round(last_price), 0.05, 1, option_type='call'))
print("Put option price: ", interface.option_pricer(prices_paths, round(last_price), 0.05, 1, option_type='put'))

for k, v in sorted(option_prices.items()):
    print(f"Strike {k}: Price {v:.3f}")
