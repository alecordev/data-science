from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump, load

boston = load_boston()

X_train, X_test, y_train, y_test = train_test_split(
    boston.data, boston.target, test_size=0.2, random_state=42
)

pipeline = Pipeline(
    [
        ("imputer", SimpleImputer()),
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression()),
    ]
)

param_grid = {
    "imputer__strategy": ["mean", "median", "most_frequent"],
}

# Use KFold cross-validation for hyperparameter tuning
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform hyperparameter tuning using GridSearchCV
grid_search = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    scoring="neg_mean_squared_error",
    cv=cv,
    verbose=1,
    n_jobs=-1,
)
grid_search.fit(X_train, y_train)

dump(grid_search.best_estimator_, "best_model.joblib")
best_model = load("best_model.joblib")

# Use the best model to make predictions on new data
new_observation = [
    [0.1, 20.0, 7.0, 0, 0.538, 6.0, 95.0, 4.0, 2.0, 300.0, 15.3, 396.0, 4.98]
]
prediction = best_model.predict(new_observation)
print(f"Prediction for new observation: {prediction}")

# Evaluate the model using the test set
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print(f"Mean squared error: {mse}")
print(f"R2 score: {r2}")

# Store the models trained and their respective metrics, parameters and hyperparameters in a dictionary
model_dict = {
    "best_model": {
        "model": best_model,
        "params": grid_search.best_params_,
        "mse": mse,
        "r2": r2,
    }
}

print(model_dict)
