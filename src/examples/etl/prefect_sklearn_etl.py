"""
Using the good old prefect 1

    pip install "prefect>=1,<2"
"""

from prefect import task, Flow
from prefect.schedules import Schedule
from prefect.schedules.clocks import CronClock
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from joblib import dump
import pandas as pd
import os

# Define the schedule to run the flow every day at 5 AM UTC
schedule = Schedule(clocks=[CronClock("0 5 * * *")])


@task
def collect_data():
    # Load the iris dataset
    iris = load_iris()
    # Convert to a Pandas DataFrame
    data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    # Add target column to DataFrame
    data["target"] = iris.target
    data["feature_names"] = iris.feature_names
    return data


@task(nout=4)
def prepare_data(data):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        data[data["feature_names"]], data["target"], test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


@task
def train_model(X_train, y_train):
    # Train a Logistic Regression classifier
    clf = LogisticRegression(random_state=42)
    clf.fit(X_train, y_train)
    return clf


@task
def calculate_metrics(clf, X_test, y_test):
    # Calculate the accuracy of the model
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc


@task
def save_metrics(acc):
    # Save the accuracy metric to a file
    with open("accuracy.txt", "w") as f:
        f.write(str(acc))


@task
def save_model(clf):
    # Save the trained model to a file
    dump(clf, "model.joblib")


# Define the flow
with Flow("train-logistic-regression", schedule=schedule) as flow:
    data = collect_data()
    X_train, X_test, y_train, y_test = prepare_data(data)
    clf = train_model(X_train, y_train)
    acc = calculate_metrics(clf, X_test, y_test)
    save_metrics(acc)
    save_model(clf)

# Run the flow
flow.run()
