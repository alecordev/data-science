import numpy as np
import matplotlib.pyplot as plt


class SimpleLinearRegression:
    """
    A class which implements simple linear regression model.
    """

    def __init__(self):
        self.b0 = None
        self.b1 = None

    def fit(self, X, y):
        """
        Used to calculate slope and intercept coefficients.

        :param X: array, single feature
        :param y: array, true values
        :return: None
        """
        numerator = np.sum((X - np.mean(X)) * (y - np.mean(y)))
        denominator = np.sum((X - np.mean(X)) ** 2)
        self.b1 = numerator / denominator
        self.b0 = np.mean(y) - self.b1 * np.mean(X)

    def predict(self, X):
        """
        Makes predictions using the simple line equation.

        :param X: array, single feature
        :return: None
        """
        if not self.b0 or not self.b1:
            raise Exception(
                'Please call `SimpleLinearRegression.fit(X, y)` before making predictions.'
            )
        return self.b0 + self.b1 * X


def random_dataset(save=False):
    X = np.arange(start=1, stop=301)
    y = np.random.normal(loc=X, scale=20)

    if save:
        plt.scatter(X, y, s=200, c='#087E8B', alpha=0.65)
        plt.title('Source dataset', size=20)
        plt.xlabel('X', size=14)
        plt.ylabel('Y', size=14)
        plt.savefig(
            '001_SimpleLinearRegression_source_dataset.png',
            dpi=300,
            bbox_inches='tight',
        )
        plt.show()

    return X, y


def visualize_best_fit_line(X, y):
    model_all = SimpleLinearRegression()
    model_all.fit(X, y)
    preds_all = model_all.predict(X)

    plt.scatter(X, y, s=200, c='#087E8B', alpha=0.65, label='Source data')
    plt.plot(
        X,
        preds_all,
        color='#000000',
        lw=3,
        label=f'Best fit line > B0 = {model_all.b0:.2f}, B1 = {model_all.b1:.2f}',
    )
    plt.title('Best fit line', size=20)
    plt.xlabel('X', size=14)
    plt.ylabel('Y', size=14)
    plt.legend()
    plt.show()


def testing_linear_regression():
    X, y = random_dataset()

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = SimpleLinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print(model.b0, model.b1)

    from sklearn.metrics import mean_squared_error

    rmse = lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred))
    print(rmse(y_test, preds))

    from sklearn.linear_model import LinearRegression

    sk_model = LinearRegression()
    sk_model.fit(np.array(X_train).reshape(-1, 1), y_train)
    sk_preds = sk_model.predict(np.array(X_test).reshape(-1, 1))

    print(sk_model.intercept_, sk_model.coef_)
    rmse(y_test, sk_preds)


if __name__ == '__main__':
    testing_linear_regression()
