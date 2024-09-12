import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xgboost as xgb
from xgboost import XGBClassifier

from sklearn.model_selection import (
    KFold,
    train_test_split,
    RandomizedSearchCV,
    cross_validate,
)
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

import joblib
from scipy.stats import uniform, randint

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_data(filepath):
    try:
        data = pd.read_csv(filepath)
        X = data.drop(["transaction_id", "customer_id", "fraud"], axis=1)
        y = data["fraud"]
        return X, y
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise


def preprocess_data(X):
    categorical_cols = X.select_dtypes(include=["object"]).columns
    X[categorical_cols] = X[categorical_cols].astype("category")
    return X


def create_pipeline():
    numeric_features = ["amount"]
    categorical_features = ["merchant", "location", "transaction_type"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "clf",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    n_jobs=-1,
                    # early_stopping_rounds=10,
                ),
            ),
        ]
    )

    return pipeline


def create_random_search(pipeline, X, y):
    param_dist = {
        "clf__n_estimators": randint(50, 200),
        "clf__max_depth": randint(3, 10),
        "clf__learning_rate": uniform(0.01, 0.3),
        "clf__subsample": uniform(0.5, 0.5),
        "clf__colsample_bytree": uniform(0.5, 0.5),
    }

    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=50,
        cv=5,
        scoring="roc_auc",
        verbose=1,
        n_jobs=-1,
        random_state=42,
    )

    return random_search


def cross_validate_model(pipeline, X, y):
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    scores = cross_validate(
        pipeline, X, y, cv=kfold, scoring=scoring, return_train_score=False, n_jobs=-1
    )

    logging.info("Cross-Validation Results (Mean Scores):")
    for metric in scoring:
        logging.info(f"{metric.capitalize()}: {np.mean(scores[f'test_{metric}']):.4f}")

    pipeline.fit(X, y)
    return pipeline


def save_model(model, filepath):
    joblib.dump(model, filepath)
    logging.info(f"Model saved to {filepath}")


def load_model(filepath):
    if not os.path.exists(filepath):
        logging.error(f"Model file not found: {filepath}")
        raise FileNotFoundError(f"Model file not found: {filepath}")
    return joblib.load(filepath)


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    logging.info(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    logging.info(f"Precision: {precision_score(y_test, y_pred):.4f}")
    logging.info(f"Recall: {recall_score(y_test, y_pred):.4f}")
    logging.info(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    logging.info(f"ROC AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

    xgb_model = model.named_steps["clf"]
    xgb.plot_importance(xgb_model)
    plt.savefig("feature_importance.png")
    plt.close()


def infer_single_observation(model, observation):
    df = pd.DataFrame(
        [observation], columns=["amount", "merchant", "location", "transaction_type"]
    )
    df = preprocess_data(df)
    prediction = model.predict(df)
    probability = model.predict_proba(df)[0][1]
    logging.info(f"Prediction for the observation: {prediction[0]}")
    logging.info(f"Probability of fraud: {probability:.4f}")
    return prediction[0]


def main_training_pipeline(data_filepath, model_filepath):
    X, y = load_data(data_filepath)
    X = preprocess_data(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = create_pipeline()
    best_model = cross_validate_model(pipeline, X_train, y_train)
    save_model(best_model, model_filepath)
    evaluate_model(best_model, X_test, y_test)


def main_inference_pipeline(model_filepath, observation):
    model = load_model(model_filepath)
    infer_single_observation(model, observation)


if __name__ == "__main__":
    data_filepath = "fraud_detection_data.csv"
    model_filepath = "fraud_detection_xgb_model.pkl"
    observation = [220.00, "Walmart", "USA", "in-store"]

    main_training_pipeline(data_filepath, model_filepath)
    main_inference_pipeline(model_filepath, observation)
