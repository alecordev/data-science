import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
)

# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import xgboost


def now():
    return datetime.datetime.utcnow()


def log(msg):
    print(f"[{now().isoformat()}] {msg}")


def load_data():
    df = pd.read_csv(
        "https://raw.githubusercontent.com/alecordev/data-science/master/data/iris.csv"
    )
    df.columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species",
    ]
    print("Shape of the dataset: " + str(df.shape))
    return df


def split_features_and_target(df):
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    print("The independent features set: ")
    print(X[:5, :])
    print("The dependent variable: ")
    print(y[:5])
    return X, y


def clean_transform_and_prepare(X, y):
    ly = LabelEncoder()
    y = ly.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=21
    )

    minmax = MinMaxScaler(feature_range=(0, 1))
    X_train = minmax.fit_transform(X_train)
    X_test = minmax.transform(X_test)

    # sns.pairplot(
    #     df[["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]],
    #     hue="species",
    #     diag_kind="kde",
    # )
    return X_train, X_test, y_train, y_test


def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, charts=False):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=123)
    n_iter_search = 10

    clf = model[0]
    param_grid = model[1]

    gs = RandomizedSearchCV(
        clf,
        param_distributions=param_grid,
        n_iter=n_iter_search,
        cv=cv,
        scoring="accuracy",
    )
    gs.fit(X_train, y_train)
    print(f"The best parameters are {gs.best_params_}")

    y_pred = gs.best_estimator_.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")

    print("*" * 30)
    print(model)
    print(f"Confusion Matrix:\n{cm}")
    print(f"Accuracy score: {acc * 100:.04f}%")
    print(f"Precision score: {precision * 100:.04f}%")
    print(f"Recall score: {recall * 100:.04f}%")
    print("*" * 30)
    if charts:
        pass


def dimensionality_reduction():
    pca = PCA(n_components=2)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    cum_explained_variance = np.cumsum(pca.explained_variance_ratio_)

    markers = ("s", "x", "o")
    colors = ("red", "blue", "lightgreen")
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
            x=X_train[y_train == cl, 0],
            y=X_train[y_train == cl, 1],
            alpha=0.8,
            c=colors[idx],
            marker=markers[idx],
            label=target_names[cl],
            edgecolor="black",
        )
    plt.xlabel("PC1")
    plt.xticks([])
    plt.ylabel("PC2")
    plt.yticks([])
    plt.title(
        "2 components, captures {}% of total variation".format(
            cum_explained_variance[1].round(4) * 100
        )
    )
    plt.legend(loc="lower right")
    plt.show()


def main():
    df = load_data()
    X, y = split_features_and_target(df)
    X_train, X_test, y_train, y_test = clean_transform_and_prepare(X, y)

    # List of tuples, each tuple is the model and its corresponding params_grid to use
    # when looking for the best hyperparameters
    models = [
        (
            RandomForestClassifier(),
            {
                "n_estimators": [100, 200],
                # "max_depth": [10, 20, 100, None],
                "max_features": ["sqrt", None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4, 10],
                "bootstrap": [True, False],
                "criterion": ["gini", "entropy"],
                "max_depth": np.arange(1, 20, 2),
            },
        ),
        (
            LogisticRegression(solver="lbfgs", multi_class="auto"),
            {
                "penalty": ["l2"],
                "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
            },
        ),
        (GaussianNB(), {"priors": [None]}),
        # SVC(C=50, kernel="rbf", gamma=1),
        (
            SVC(kernel="rbf", probability=True),
            {"gamma": np.logspace(-2, 2, 5), "C": np.logspace(-2, 2, 5)},
        ),
        (
            KNeighborsClassifier(n_neighbors=3, algorithm="ball_tree"),
            {
                "n_neighbors": np.arange(1, 15),
                "weights": ["uniform", "distance"],
                "leaf_size": [1, 3, 5],
                "algorithm": ["auto", "kd_tree"],
            },
        ),
        (
            DecisionTreeClassifier(),
            {
                "criterion": ["gini", "entropy"],
                "splitter": ["best", "random"],
                "max_depth": np.arange(1, 20, 2),
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4, 10],
                "max_features": ["sqrt", "log2", None],
            },
        ),
        (
            xgboost.XGBClassifier(
                objective="binary:logistic", max_depth=4, n_estimators=10
            ),
            {
                # "silent": [False],
                "max_depth": [6, 10, 15, 20],
                "learning_rate": [0.001, 0.01, 0.1, 0.2, 0, 3],
                "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                "colsample_bylevel": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                "min_child_weight": [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],
                "gamma": [0, 0.25, 0.5, 1.0],
                "reg_lambda": [0.1, 1.0, 5.0, 10.0, 50.0, 100.0],
                "n_estimators": [100],
            },
        ),
        (
            QuadraticDiscriminantAnalysis(),
            {"priors": [None], "reg_param": np.arange(0.0, 1.0, 0.1)},
        ),
    ]
    for m in models:
        train_and_evaluate_model(m, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
