import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
data = pd.read_csv("example_dataset.csv")
X = data.drop("target_column", axis=1)
y = data["target_column"]

# Create a pipeline
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

preprocessor = ColumnTransformer(transformers=[("num", numeric_transformer, X.columns)])

model_pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("model", XGBRegressor())]
)

# Define the hyperparameter space to search over
param_grid = {
    "model__learning_rate": [0.01, 0.1],
    "model__max_depth": [3, 5, 7],
    "model__n_estimators": [100, 200, 300],
}

# Set up the cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform the grid search with cross-validation
grid_search = GridSearchCV(
    model_pipeline,
    param_grid,
    cv=cv,
    verbose=2,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
)
grid_search.fit(X, y)

# Get the best model and its metrics
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X)

mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("Best Parameters: ", grid_search.best_params_)
print("Best Model Score: ", grid_search.best_score_)
print("MSE: ", mse)
print("R2 Score: ", r2)

# Save the best model
from joblib import dump

dump(best_model, "xgboost_regression_model.joblib")
