# Import libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import plotly.graph_objs as go
import statsmodels.api as sm
# Tools for stationarity tests and VAR models
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.var_model import VAR
# To download financial data
import yfinance as yf 
# To evaluate forecast quality
from sklearn.metrics import mean_squared_error

# 1. Data Download 
# Download monthly data for BNP Paribas and VGK ETF between 2014 and 2019
bnp = yf.download('BNP.PA', start='2014-01-01', end='2019-12-31', interval='1mo')
vgk = yf.download('VGK', start='2014-01-01', end='2019-12-31', interval='1mo')
# Concatenate 'Close' columns of both series, aligning them on common dates
data = pd.concat([bnp['Close'], vgk['Close']], axis=1)
data.columns = ['BNP', 'VGK']
data = data.dropna()

# 2. Logarithmic transformation and return calculation
log_data = np.log(data)
# Calculate log returns: difference of logs
returns = log_data.diff().dropna()

# 3. ADF Stationarity Test
# Function to perform ADF test and print results
def run_adf(series, name):
    result = adfuller(series)
    print(f"\nADF Test for {name}")
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    print("Conclusion:", "Stationary" if result[1] < 0.05 else "Non-stationary")

# Apply ADF test on returns
run_adf(returns['BNP'], "BNP")
run_adf(returns['VGK'], "VGK")

# ================================
# 4. VAR Model Estimation
# ================================

# Create a VAR model from log returns
model = VAR(returns)

# Estimate model with 2 lags (order p=2)
results = model.fit(2)

# Display VAR model summary
print("\nVAR Model Summary:")
print(results.summary())

# ================================
# 5. Forecasting with a 2-period horizon
# ================================

# Use the last 2 observations for forecasting
forecast = results.forecast(returns.values[-2:], steps=2)

# Store results in a DataFrame for readability
forecast_df = pd.DataFrame(forecast, columns=['BNP_forecast', 'VGK_forecast'])
print("\nLog returns forecast (2 steps):")
print(forecast_df)

# ================================
# 6. Conversion to price levels
# ================================

# Last observed prices
last_prices = data.iloc[-1]

# Accumulate forecasted returns to get log prices
predicted_log_returns = forecast_df.cumsum()

# Convert back to prices (exponential of log)
predicted_prices = last_prices.values * np.exp(predicted_log_returns)

# Display forecasted prices
print("\nForecasted Prices:")
print(pd.DataFrame(predicted_prices, columns=data.columns))

# ================================
# 7. (Optional) Evaluation with RMSE
# ================================

# If you have actual future values, compare here
# rmse_bnp = mean_squared_error(y_true, y_pred_bnp, squared=False)
# rmse_vgk = mean_squared_error(y_true, y_pred_vgk, squared=False)

# ================================
# 8. Impulse Response Functions (IRF)
# ================================

# Calculate IRFs over 10 periods
irf = results.irf(10)

# Plot IRFs
irf.plot(orth=False)
plt.tight_layout()
plt.show()
