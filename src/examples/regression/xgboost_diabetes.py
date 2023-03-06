import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib

# Load the Boston dataset
diabetes = load_diabetes()

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data, diabetes.target, test_size=0.2, random_state=42
)

# Define the pipeline
pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", xgb.XGBRegressor(random_state=42)),
    ]
)

# Define the hyperparameters for tuning
param_distributions = {
    "model__n_estimators": np.arange(50, 500, 50),
    "model__learning_rate": np.logspace(-3, 0, 4),
    "model__max_depth": np.arange(2, 10, 2),
    "model__min_child_weight": np.arange(1, 10, 2),
    "model__subsample": np.arange(0.6, 1.0, 0.1),
    "model__colsample_bytree": np.arange(0.6, 1.0, 0.1),
    "model__reg_alpha": np.logspace(-5, 1, 7),
    "model__reg_lambda": np.logspace(-5, 1, 7),
}

# Perform a randomized search with cross-validation
search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_distributions,
    cv=5,
    n_iter=50,
    n_jobs=-1,
    random_state=42,
)
search.fit(X_train, y_train)

# Store the best model and its hyperparameters
best_model = search.best_estimator_
best_params = search.best_params_

# Save the best model to disk
joblib.dump(best_model, "xgboost_boston.joblib")

# Load the best model from disk
loaded_model = joblib.load("xgboost_boston.joblib")

# Use the loaded model to make predictions on a new observation
new_observation = [[0.01778, 95, 1.47, 0, 0.403, 7.135, 13.9, 7.6534, 5, 421]]
prediction = loaded_model.predict(new_observation)
print(f"Predicted price: {prediction[0]}")

# Evaluate the model on the test set
y_pred = loaded_model.predict(X_test)
metrics = {
    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
    "MAE": mean_absolute_error(y_test, y_pred),
    "R2": r2_score(y_test, y_pred),
}
print("Model evaluation metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.3f}")
