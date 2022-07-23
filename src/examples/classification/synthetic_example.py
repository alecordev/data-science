import datetime

from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def now():
    return datetime.datetime.utcnow()


def log(msg):
    print(f"[{now().isoformat()}] {msg}")


def main():
    X, y = datasets.make_classification(
        n_samples=1000,
        n_features=64,
        n_classes=2,
        n_redundant=0,
        n_clusters_per_class=1,
        weights=[0.7, 0.3],
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    rf = RandomForestClassifier(random_state=42)
    params_rf = {
        "n_estimators": [200, 300, 400],
        "max_depth": [4, 6, 8],
        "min_samples_leaf": [0.1, 0.2],
        "max_features": ["log2", "sqrt"],
    }

    grid_rf = GridSearchCV(
        estimator=rf,
        param_grid=params_rf,
        cv=10,
        scoring="accuracy",
        verbose=1,
        n_jobs=-1,
    )
    grid_rf.fit(X_train, y_train)
    best_params = grid_rf.best_params_
    log(f"Best parameters for the model: {best_params}")

    predictions_test = grid_rf.predict_proba(X_test)[:, 1]

    auc_score = roc_auc_score(y_test, predictions_test)
    log(f"AUC Score: {auc_score}")

    best_model = grid_rf.best_estimator_
    test_acc = best_model.score(X_test, y_test)

    log(f"Test set accuracy of best model: {test_acc:.3f}")
    log(
        f"Probability of class 1 in observation (row) 0: {grid_rf.predict_proba(X[0].reshape(1, -1))[:, 1]}"
    )  # prob of class 1


main()
