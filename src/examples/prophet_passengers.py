"""
pandas==1.0.5
pystan==2.19.1.1
fbprophet==0.6
"""
import pandas as pd
from fbprophet import Prophet

import matplotlib.pyplot as plt

plt.style.use("fivethirtyeight")


def process():
    url = (
        "https://assets.digitalocean.com/articles/eng_python/prophet/AirPassengers.csv"
    )
    df = pd.read_csv(url)
    df["Month"] = pd.DatetimeIndex(df["Month"])
    df = df.rename(columns={"Month": "ds", "AirPassengers": "y"})

    ax = df.set_index("ds").plot(figsize=(12, 8))
    ax.set_ylabel("Monthly Number of Airline Passengers")
    ax.set_xlabel("Date")
    plt.savefig(fname="raw_data.png")

    # set the uncertainty interval to 95% (the Prophet default is 80%)
    model = Prophet(interval_width=0.95)
    model.fit(df)
    future_dates = model.make_future_dataframe(periods=36, freq="MS")
    forecast = model.predict(future_dates)
    model.plot(forecast, uncertainty=True).savefig("forecast.png")
    model.plot_components(forecast).savefig("components.png")


if __name__ == "__main__":
    process()
