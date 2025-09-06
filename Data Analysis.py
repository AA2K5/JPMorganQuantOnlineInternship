import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

df = pd.read_csv('Nat_Gas.csv')
df["Dates"] = pd.to_datetime(df["Dates"])
df.set_index('Dates', inplace= True)

df['Prices'].plot(kind = 'line', title = 'Prices over Time')
plt.xlabel("Date")
plt.ylabel("Prices")
plt.grid(True)
plt.tight_layout
plt.show()


# From results, note that the prices tend to increase from June to Jan and decrease from Jan to June, with an overall upwards trend to prices. 

from statsmodels.tsa.seasonal import seasonal_decompose

decomp = seasonal_decompose(df['Prices'], model='addition', period=12)
decomp.plot()
plt.show()

# This model confirms result, with small residuals and constant variance over time, confirming homoskedastic properties

# Building a Linear Regression Model on this:
trend = decomp.trend
seasonal = decomp.seasonal
resid = decomp.resid

from sklearn.linear_model import LinearRegression
trend = trend.dropna()
X = np.arange(len(trend)).reshape(-1,1)
y = trend.values

model = LinearRegression().fit(X,y)

future_periods = 12
X_future = np.arange(len(trend), len(trend) + future_periods).reshape(-1,1)
forecast = model.predict(X_future)

seasonal_pattern = seasonal[:12].values
seasonal_forecast = np.tile(seasonal_pattern, future_periods // 12 + 1)[:future_periods]

price_forecast = forecast + seasonal_forecast

future_dates = pd.date_range(start=df.index[-1] + pd.offsets.MonthBegin(1), periods=future_periods, freq='MS')

plt.figure(figsize=(10, 5))
plt.plot(df.index, df['Prices'], label='Observed')
plt.plot(future_dates, price_forecast, label='Forecast', linestyle='--')
plt.title('Forecast Using Seasonal Decomposition')
plt.xlabel('Date')
plt.ylabel('Prices')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

def estimate_price(input_date):
    input_date = pd.to_datetime(input_date)
    if input_date in df.index:
        return f"Exact price on {input_date.date()}: ${df.loc[input_date, 'Prices']:.2f}"
    elif input_date < df.index[-1]:
        estimated = df['Prices'].reindex(df.index.union([input_date])).sort_index().interpolate(method='time')
        return f"Interpolated price on {input_date.date()}: ${estimated.loc[input_date]:.2f}"
    else:
        months_ahead = (input_date.year - future_dates[0].year) * 12 + (input_date.month - future_dates[0].month)
        if months_ahead < 0 or months_ahead >= len(price_forecast):
            return "Date is too far outside forecast range."
        else:
            forecasted_price = price_forecast[months_ahead]
            return f"Forecasted price on {input_date.date()}: ${forecasted_price:.2f}"

print(estimate_price("2023-04-01"))  # Past (interpolated)
print(estimate_price("2025-06-01"))  # Future (forecasted)