import json

from sklearn.datasets import load_iris
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    RandomizedSearchCV,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
from xgboost import XGBClassifier
import joblib

iris = load_iris()

# # Create a dataframe from the data
# data = pd.DataFrame(iris.data, columns=iris.feature_names)

# # Add the target variable to the dataframe
# data['target'] = iris.target

# Split into features and labels
X, y = iris.data, iris.target

le = LabelEncoder()
y = le.fit_transform(y)

pipe = Pipeline(
    [
        ("imputer", SimpleImputer()),
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(objective="multi:softmax")),
    ]
)

# Define hyperparameters for tuning
hyperparameters = {
    "imputer__strategy": ["mean", "median", "most_frequent"],
    "clf__max_depth": [3, 4, 5],
    "clf__n_estimators": [50, 100, 150, 200, 500],
    "clf__learning_rate": [0.01, 0.1, 0.5],
    "clf__subsample": [0.5, 0.8, 1.0],
    "clf__colsample_bytree": [0.5, 0.8, 1.0],
    "clf__reg_alpha": [0, 0.1, 0.5, 1],
    "clf__reg_lambda": [0, 0.1, 0.5, 1],
    "clf__num_class": [3],
}

# Use k-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search_flag = False

if grid_search_flag:
    # Perform hyperparameter tuning using GridSearchCV
    grid_search = GridSearchCV(pipe, param_grid=hyperparameters, cv=cv, n_jobs=-1)
    grid_search.fit(X, y)

    # Save the model
    joblib.dump(grid_search.best_estimator_, "iris_xgb_model.pkl")
else:
    # Create a randomized search object with 5-fold cross-validation
    random_search = RandomizedSearchCV(
        pipe, hyperparameters, cv=5, n_iter=50, n_jobs=-1, random_state=42
    )

    # Fit the randomized search object to the data
    random_search.fit(X, y)

    # Print the best parameters and score
    print("Best parameters: ", random_search.best_params_)
    print("Best score: ", random_search.best_score_)

    # Save the best model to disk
    best_model = random_search.best_estimator_
    joblib.dump(best_model, "iris_xgb_model.pkl")

# Load the model
model = joblib.load("iris_xgb_model.pkl")

# Predict on test data and calculate metrics
y_pred = model.predict(X)

# print("Classification Report:")
# print(classification_report(y, y_pred))

# cm = confusion_matrix(y, y_pred)
# tn, fp, fn, tp = cm.ravel()

report = classification_report(y, y_pred, output_dict=True)

accuracy = report["accuracy"]
precision = report["1"]["precision"]
recall = report["1"]["recall"]
f1_score = report["1"]["f1-score"]

print("Confusion Matrix:")
print(confusion_matrix(y, y_pred))

# print({"tn": tn, "fp": fp, "fn": fn, "tp": tp})
# fn_rate = fn / (fn + tp)
# fp_rate = fp / (fp + tn)

if grid_search_flag:
    best_params = grid_search.best_params_
else:
    best_params = random_search.best_params_

print(f"Parameters: {best_params}")
print(
    f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1_score:.3f}"
)
# print(f"False Negative Rate: {fn_rate:.3f}, False Positive Rate: {fp_rate:.3f}")
# print(cm)
print(json.dumps(report, indent=4))
