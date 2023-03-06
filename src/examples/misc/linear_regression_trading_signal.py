import numpy as np

from scipy import stats
from statsmodels import robust
from sklearn.linear_model import LinearRegression


def linear_regression_signal(x, y, current_read, accuracy_threshold: float = 0.85):
    """

    Parameters
    ----------
    x : list
        Sequence of pairs (tuples or lists or iterable): (value1, value2)
    y : list
        Sequence of targets
    accuracy_threshold : float
        Threshold to make a decision

    Returns
    -------

    """
    print('Computing linear regression signal...')
    # low_price, volume = x[-1]  # for today

    x = np.asarray(x)
    y = np.asarray(y)

    model = LinearRegression()
    model.fit(x, y)

    # Predict the corresponding value of Y for X
    new_feature = [current_read[0], current_read[1]]

    new_feature = np.asarray(new_feature)
    new_feature = new_feature.reshape(1, -1)
    price_prediction = model.predict(new_feature)

    score = model.score(x, y)

    current_price = current_read[0]
    price_diff = current_price - price_prediction
    print(
        f'Linear Regression price prediction: {price_prediction[0]:0.2f}. Current price: {current_price}. Accuracy: {score:0.3f}.'
    )

    if price_diff < 0 and score > accuracy_threshold:
        action = 'SELL'
    elif price_diff > 0 and score > accuracy_threshold:
        action = 'BUY'
    else:
        action = 'NO ACTION'
    print(action)
    return action


def slopes_signal(
    low_prices: list, high_prices: list, current_bid, method: str = 'MAD'
):
    """

    Parameters
    ----------
    low_prices
    high_prices
    current_bid
    method

    Returns
    -------

    """
    print('Computing slopes signal...')
    x = np.ma.asarray(low_prices)
    y = np.ma.asarray(high_prices)
    xi = np.arange(0, len(x))
    (
        low_prices_slope,
        low_prices_intercept,
        low_prices_lo_slope,
        low_prices_hi_slope,
    ) = stats.mstats.theilslopes(x, xi, 0.99)
    (
        high_prices_slope,
        high_prices_intercept,
        high_prices_lo_slope,
        high_prices_hi_slope,
    ) = stats.mstats.theilslopes(y, xi, 0.99)

    if method.upper() == 'MAD':
        var_upper = float(high_prices_intercept + (abs(robust.mad(y) * 3)))
        var_lower = float(low_prices_intercept - (abs(robust.mad(x) * 3)))
    else:  # method.upper() == 'IQR':
        var_upper = float(
            high_prices_intercept + (abs(stats.iqr(y, nan_policy='omit') * 2))
        )
        var_lower = float(
            low_prices_intercept - (abs(stats.iqr(x, nan_policy='omit') * 2))
        )

    print(f'Upper value: {var_upper}. Lower value: {var_lower}')
    if float(current_bid) > var_upper:
        action = 'BUY'
        print('BUY')
    elif float(current_bid) < var_lower:
        action = 'SELL'
        print('SELL')
    else:
        action = 'NO ACTION'
        print('Nothing to trade - yet')

    return action
