from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate, ParameterGrid
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Load breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Define pipeline
pipe = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression()),
    ]
)

# Define parameter grid
param_grid = {"clf__C": [0.01, 0.1, 1, 10, 100], "clf__penalty": ["l2"]}

# Define cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define thresholds for false negatives and false positives
fn_threshold = 0.1
fp_threshold = 0.1

# Initialize best model and metrics
best_model = None
best_fn = 1.0
best_fp = 1.0

# Iterate over parameter combinations
for params in ParameterGrid(param_grid):

    # Set parameters
    pipe.set_params(**params)

    # Cross-validate model
    results = cross_validate(
        pipe, X, y, cv=cv, scoring=["accuracy", "precision", "recall", "f1"]
    )

    # Calculate mean metrics
    acc = results["test_accuracy"].mean()
    prec = results["test_precision"].mean()
    rec = results["test_recall"].mean()
    f1 = results["test_f1"].mean()

    # Calculate false negative and false positive rates
    tn, fp, fn, tp = confusion_matrix(y, pipe.fit(X, y).predict(X)).ravel()
    fn_rate = fn / (fn + tp)
    fp_rate = fp / (fp + tn)

    # Print metrics and rates
    print(f"Parameters: {params}")
    print(
        f"Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}"
    )
    print(f"False Negative Rate: {fn_rate:.3f}, False Positive Rate: {fp_rate:.3f}")

    # Check if model is better than previous best
    if fn_rate <= best_fn and fp_rate <= best_fp:
        best_model = pipe
        best_fn = fn_rate
        best_fp = fp_rate
        print("New best model!")

        # Calculate and print confusion matrix
        y_pred = best_model.fit(X, y).predict(X)
        cm = confusion_matrix(y, y_pred)
        print(f"Confusion Matrix:\n{cm}\n")

import joblib

joblib.dump(best_model, "best_model.pkl")
