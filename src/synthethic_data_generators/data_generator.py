import random
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


def generate_data():
    categories = ["CATEGORY1", "CATEGORY2", "CATEGORY3"]
    feature1 = np.random.randn(1000)
    feature2 = feature1.cumsum()
    feature3 = feature1 * 0.95
    feature4 = feature1 * 0.2
    feature5 = feature1 * random.choice([feature2, feature3])
    feature6 = feature1 * np.random.randn(1000)

    data = pd.DataFrame(
        {
            "date": pd.date_range(start="20150101", periods=1000),
            "importance": [random.choice(list(range(10))) for _ in range(1000)],
            "col1": np.random.randn(1000),
            "col2": np.random.randn(1000),
            "col3": np.random.randn(1000),
            "num": [np.random.randint(1, 10) for _ in range(1000)],
            "num2": [np.random.randint(10, 20) for _ in range(1000)],
            "initial": np.random.uniform(low=15000000, high=500000000, size=(1000,)),
            "final": np.random.uniform(low=10000000, high=400000000, size=(1000,)),
            "size": np.random.uniform(low=10e10, high=9e12, size=(1000,)),
            "dist": np.random.exponential(scale=1e10, size=(1000,)),
            "normal": np.random.normal(scale=1.0, size=(1000,)),
            "category": [random.choice(categories) for _ in range(1000)],
            "feature1": feature1,
            "feature2": feature2,
            "feature3": feature3,
            "feature4": feature4,
            "feature5": feature5,
            "feature6": feature6,
        }
    )
    return data


def generate_classification_data(
    n_samples=1000, n_features=20, n_classes=3, random_state=42, class_weights=None
):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=random_state,
        weights=class_weights,
    )
    # Optionally scale the features for better machine learning performance
    X = StandardScaler().fit_transform(X)
    return pd.DataFrame(X, columns=[f"feature_{i+1}" for i in range(n_features)]), y


def generate_regression_data(n_samples=1000, n_features=20, random_state=42, noise=0.1):
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=random_state,
    )
    # Optionally scale the features for better machine learning performance
    X = StandardScaler().fit_transform(X)
    return pd.DataFrame(X, columns=[f"feature_{i+1}" for i in range(n_features)]), y


def generate_imbalanced_classification_data(
    n_samples=1000,
    n_features=20,
    n_classes=3,
    class_weights=[0.1, 0.3, 0.6],
    random_state=42,
):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=random_state,
        weights=class_weights,
    )
    # Optionally scale the features for better machine learning performance
    X = StandardScaler().fit_transform(X)
    return pd.DataFrame(X, columns=[f"feature_{i+1}" for i in range(n_features)]), y


def generate_classification_with_overlap(
    n_samples=1000, n_features=2, n_classes=3, overlap=0.5, random_state=42
):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_clusters_per_class=1,
        class_sep=overlap,
        random_state=random_state,
    )
    X = StandardScaler().fit_transform(X)
    return pd.DataFrame(X, columns=[f"feature_{i+1}" for i in range(n_features)]), y


def add_noise_to_data(X, noise_factor=0.1):
    noise = np.random.normal(0, noise_factor, X.shape)
    return X + noise


def main():
    print("Generated Basic Data Example:")
    basic_data = generate_data()
    print(basic_data.head(), "\n")

    X_class, y_class = generate_classification_data(
        n_samples=500, n_features=10, n_classes=3
    )
    print("Generated Classification Dataset:")
    print(X_class.head(), y_class[:5], "\n")

    X_reg, y_reg = generate_regression_data(n_samples=500, n_features=10)
    print("Generated Regression Dataset:")
    print(X_reg.head(), y_reg[:5], "\n")

    X_imbalance, y_imbalance = generate_imbalanced_classification_data(
        n_samples=500, n_features=10, class_weights=[0.2, 0.3, 0.5]
    )
    print("Generated Imbalanced Classification Dataset:")
    print(X_imbalance.head(), y_imbalance[:5], "\n")

    X_overlap, y_overlap = generate_classification_with_overlap(
        n_samples=500, n_features=10, overlap=0.3
    )
    print("Generated Classification Dataset with Overlap:")
    print(X_overlap.head(), y_overlap[:5], "\n")

    X_noisy = add_noise_to_data(X_class, noise_factor=0.2)
    print("Generated Noisy Data (first 5 samples):")
    print(X_noisy[:5], "\n")


if __name__ == "__main__":
    main()
