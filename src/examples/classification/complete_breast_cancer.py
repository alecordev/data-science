import json

from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

from joblib import dump, load
from tqdm import tqdm

from sklearn.exceptions import ConvergenceWarning

ConvergenceWarning("ignore")


def run_pipeline(thresholds):
    data = load_breast_cancer()
    X = data.data
    y = data.target

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(penalty="l2")),
        ]
    )

    param_grid = {
        "model__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        "model__solver": ["lbfgs", "liblinear"],  # , "saga"],
        "model__max_iter": [1000, 2000, 5000],
    }

    # create stratified k-fold for cross validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # create grid search object
    grid_search = GridSearchCV(
        pipeline, param_grid, scoring="accuracy", cv=cv, n_jobs=-1, verbose=1
    )

    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X)

    # get the confusion matrix and classification report
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    report = classification_report(y, y_pred, output_dict=True)

    accuracy = report["accuracy"]
    precision = report["1"]["precision"]
    recall = report["1"]["recall"]
    f1_score = report["1"]["f1-score"]

    print({"tn": tn, "fp": fp, "fn": fn, "tp": tp})
    fn_rate = fn / (fn + tp)
    fp_rate = fp / (fp + tn)

    # Print metrics and rates
    print(f"Parameters: {grid_search.best_params_}")
    print(
        f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1_score:.3f}"
    )
    print(f"False Negative Rate: {fn_rate:.3f}, False Positive Rate: {fp_rate:.3f}")
    print(cm)
    print(json.dumps(report, indent=4))
    print("Evaluation metrics for the best model:")
    print(classification_report(y, best_model.predict(X)))

    # set thresholds and calculate false positives and false negatives
    # fp_thresh = thresholds.get("false_positive", 0.2)
    # fn_thresh = thresholds.get("false_negative", 0.2)

    # fp_count = 0
    # fn_count = 0

    # for i in tqdm(range(len(y))):
    #     if y[i] == 0 and y_pred[i] == 1 and report["0"]["recall"] < fn_thresh:
    #         fn_count += 1
    #     elif y[i] == 1 and y_pred[i] == 0 and report["1"]["recall"] < fp_thresh:
    #         fp_count += 1

    # store the best model, parameters and metrics in a dictionary
    model_dict = {
        "model": best_model,
        "params": grid_search.best_params_,
        "metrics": report,
        "confusion_matrix": cm,
        # "false_positives": fp_count,
        # "false_negatives": fn_count,
    }

    return model_dict


# set the false positive and false negative thresholds
thresholds = {"false_positive": 0.1, "false_negative": 0.1}

# run the pipeline
model_dict = run_pipeline(thresholds)
print(json.dumps(model_dict, indent=4, default=str))


def save():
    dump(model_dict["model"], "best_model.joblib")


def load():
    loaded_model = load("best_model.joblib")
