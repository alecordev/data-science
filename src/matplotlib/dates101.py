import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

np.random.seed(42)

datelist = pd.date_range(
    pd.datetime(2017, 1, 1).strftime('%Y-%m-%d'), periods=93
).tolist()
df = pd.DataFrame(
    np.cumsum(np.random.randn(93)), columns=['error'], index=pd.to_datetime(datelist)
)

plt.bar(df.index, df["error"].values)
plt.gca().xaxis.set_major_locator(mdates.DayLocator((1, 15)))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))
plt.gcf().autofmt_xdate()
plt.show()
