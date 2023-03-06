import json
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

data = load_breast_cancer()
X = data.data
y = data.target

imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()
model = LogisticRegression()

pipeline = Pipeline([("imputer", imputer), ("scaler", scaler), ("model", model)])

false_neg_threshold = 0.05
false_pos_threshold = 0.05

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

best_model = None
best_false_negs = 1.0

for train_index, test_index in skf.split(X, y):
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    false_negs = report["0"]["recall"]
    if false_negs < best_false_negs:
        best_false_negs = false_negs
        best_model = pipeline
    print(f"False negatives: {false_negs}")
    print(f"Classification report:\n{report}\n")

if best_model is not None:
    print("Best model found, saving to disk...")
    joblib.dump(best_model, "best_model.pkl")

    # Load the model to ensure it was saved correctly
    loaded_model = joblib.load("best_model.pkl")

    # Test the loaded model
    y_pred = loaded_model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    print(f"Loaded model classification report:\n{json.dumps(report, indent=4)}\n")

else:
    print("No best model found.")
