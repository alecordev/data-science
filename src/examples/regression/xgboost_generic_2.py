import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("example_dataset.csv")

# Split the data into training and testing sets
train_df = df.sample(frac=0.7, random_state=42)
test_df = df.drop(train_df.index)

# Separate the target variable from the input features
X_train = train_df.drop(["target_variable"], axis=1)
X_test = test_df.drop(["target_variable"], axis=1)
y_train = train_df["target_variable"]
y_test = test_df["target_variable"]

# Define the preprocessing steps
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

# Define the column transformer
preprocessor = ColumnTransformer(
    transformers=[("num", numeric_transformer, X_train.columns)]
)

# Define the XGBoost model
xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)

# Define the full pipeline
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("xgb_model", xgb_model)])

# Define the hyperparameters to tune
param_grid = {
    "xgb_model__n_estimators": [100, 500],
    "xgb_model__learning_rate": [0.01, 0.1],
    "xgb_model__max_depth": [3, 5, 7],
    "xgb_model__min_child_weight": [1, 3, 5],
    "xgb_model__subsample": [0.5, 0.75, 1],
    "xgb_model__colsample_bytree": [0.5, 0.75, 1],
}

# Define the cross-validation strategy
cv = 5

# Tune the hyperparameters using GridSearchCV
grid_search = GridSearchCV(
    pipeline, param_grid=param_grid, cv=cv, n_jobs=-1, scoring="neg_mean_squared_error"
)
grid_search.fit(X_train, y_train)

# Print the best hyperparameters and the corresponding mean squared error
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {-1*grid_search.best_score_}")

# Make predictions on the test set
y_pred = grid_search.predict(X_test)

# Evaluate the model using metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean squared error: {mse:.2f}")
print(f"R^2 score: {r2:.2f}")

# Save the best model
xgb_model = grid_search.best_estimator_["xgb_model"]
xgb_model.save_model("xgb_model.bin")
