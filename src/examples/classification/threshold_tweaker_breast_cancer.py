from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
)
from joblib import dump, load

# load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# define pipeline
pipe = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(penalty="l2")),
    ]
)

# define parameter grid for hyperparameter tuning
param_grid = {
    "model__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    "model__solver": ["lbfgs", "liblinear"],  # , "saga"],
    "model__max_iter": [1000, 2000, 5000],
}

# define evaluation metrics
eval_metrics = {
    "confusion_matrix": confusion_matrix,
    "accuracy": accuracy_score,
    "recall": recall_score,
    "precision": precision_score,
    "f1_score": f1_score,
}

# define thresholds for false positives and false negatives
fp_threshold = 0.05
fn_threshold = 0.05

# define stratified cross validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# perform hyperparameter tuning with grid search
grid = GridSearchCV(pipe, param_grid=param_grid, cv=cv, scoring="recall")
grid.fit(X, y)

# get best model and associated metrics
best_model = grid.best_estimator_
best_params = grid.best_params_
best_metrics = {}

for metric_name, metric_fn in eval_metrics.items():
    if metric_name == "confusion_matrix":
        best_metrics[metric_name] = metric_fn(y, best_model.predict(X))
    else:
        best_metrics[metric_name] = metric_fn(y, best_model.predict(X))

print("Best hyperparameters:", best_params)

for metric_name, metric_value in best_metrics.items():
    print(metric_name, ":", metric_value)


def save_model(best_model):
    dump(best_model, "best_model.joblib")


def predict_with_model(model_path, data):
    model = load(model_path)
    return model.predict(data)


def evaluate_model(model, data, targets, fp_threshold=0.05, fn_threshold=0.05):
    """define function to evaluate model performance with custom thresholds"""
    y_pred = model.predict(data)
    tn, fp, fn, tp = confusion_matrix(targets, y_pred).ravel()
    false_positive_rate = fp / (fp + tn)
    false_negative_rate = fn / (fn + tp)
    if false_positive_rate > fp_threshold or false_negative_rate > fn_threshold:
        print("Model did not meet thresholds for false positives or false negatives.")
    else:
        print("Model met thresholds for false positives and false negatives.")
    print("Confusion matrix:")
    print(confusion_matrix(targets, y_pred))
    print("Accuracy:", accuracy_score(targets, y_pred))
    print("Recall:", recall_score(targets, y_pred))
    print("Precision:", precision_score(targets, y_pred))
    print("F1 score:", f1_score(targets, y_pred))


# load saved model and evaluate with custom thresholds
save_model(best_model=best_model)
loaded_model = load("best_model.joblib")
evaluate_model(loaded_model, X, y, fp_threshold, fn_threshold)
