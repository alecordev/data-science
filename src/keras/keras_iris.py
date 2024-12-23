import pathlib

import numpy as np
import pandas as pd

import tensorflow as tf
from keras import layers
from keras.models import Sequential, load_model
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical


dataset_path = (
    pathlib.Path(__file__).resolve().parent.parent.parent.joinpath("data", "iris.csv")
)


def check_gpu():
    print(tf.__version__)
    physical_devices = tf.config.list_physical_devices("GPU")
    if len(physical_devices) > 0:
        print("GPU is available")
        # Optionally set memory growth to avoid allocation of all GPU memory at once
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    else:
        print("No GPU available, using CPU")


def load_data(file_path):
    df = pd.read_csv(file_path)
    feature_cols = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
    target_col = "Species"

    # Features and target
    X = df[feature_cols].values
    y = LabelEncoder().fit_transform(df[target_col].values)

    return X, y


def deepml_model(input_dim=4, num_classes=3):
    model = Sequential()
    model.add(layers.Dense(8, input_dim=input_dim, activation="relu"))
    model.add(layers.Dense(num_classes, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return model


def run_kfold_cross_validation(X, y, model_fn, n_splits=10, epochs=200, batch_size=5):
    # One-hot encode the labels (important for multi-class classification)
    dummy_y = to_categorical(y)  # Fix: Use to_categorical from keras.utils

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Initialize results container
    results = []

    # Iterate over each fold
    for train_idx, test_idx in kfold.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = dummy_y[train_idx], dummy_y[test_idx]

        # Build the model
        model = model_fn()

        # Train the model
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

        # Evaluate the model on the test fold
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        results.append(accuracy)

    # Print the final cross-validation results
    mean_accuracy = np.mean(results)
    std_accuracy = np.std(results)
    print(f"Model: {mean_accuracy * 100:.2f}% (+/- {std_accuracy * 100:.2f}%)")

    model.save("iris_model.keras")
    print("Model saved to iris_model.keras")


def load_trained_model(model_path="iris_model.keras"):
    try:
        model = load_model(model_path)
        print(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


# Function to perform inference on new data
def perform_inference(model, X_new):
    predictions = model.predict(X_new)
    predicted_classes = np.argmax(predictions, axis=1)
    print(f"Predicted classes: {predicted_classes}")
    return predicted_classes


def main():
    X, y = load_data(dataset_path)

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print(X[:5])
    print(y[:5])

    run_kfold_cross_validation(X, y, deepml_model)
    model = load_trained_model("iris_model.keras")

    if model:
        # Perform inference on an example (using first 5 samples as an example)
        print("Performing inference on first 5 samples:")
        predictions = perform_inference(model, X[:5])


# Run the main function
if __name__ == "__main__":
    check_gpu()
    main()
