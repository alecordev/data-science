from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
from joblib import dump, load
import numpy as np

# Load the diabetes dataset
diabetes = load_diabetes()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data, diabetes.target, test_size=0.2, random_state=42
)

# Define the pipeline
pipeline = Pipeline(
    [("imputer", SimpleImputer()), ("scaler", StandardScaler()), ("lasso", Lasso())]
)

# Define the hyperparameter grid
param_grid = {"lasso__alpha": np.logspace(-4, 4, 20)}

# Define the evaluation metrics
scoring = {"r2": make_scorer(r2_score), "mse": make_scorer(mean_squared_error)}

# Define the cross-validation strategy
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Define the grid search
grid_search = GridSearchCV(
    pipeline, param_grid=param_grid, cv=cv, scoring=scoring, refit="r2"
)

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# Print the best hyperparameters and metrics
print("Best hyperparameters:", grid_search.best_params_)
print("Best r2 score:", grid_search.best_score_)
print("Best MSE:", grid_search.cv_results_["mean_test_mse"][grid_search.best_index_])

# Save the best model
dump(grid_search.best_estimator_, "diabetes_model.joblib")

# Load the best model and use it to predict a new observation
loaded_model = load("diabetes_model.joblib")
new_observation = X_test[0, :].reshape(1, -1)
print("Predicted value:", loaded_model.predict(new_observation)[0])
