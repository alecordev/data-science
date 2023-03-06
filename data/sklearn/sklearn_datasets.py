from sklearn.datasets import (
    load_boston,
    load_iris,
    load_diabetes,
    load_wine,
    load_breast_cancer,
)
import pandas as pd

# Load datasets
datasets = [
    load_boston(),
    load_iris(),
    load_diabetes(),
    load_wine(),
    load_breast_cancer(),
]

# Save each dataset to CSV
for dataset in datasets:
    data = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    target = pd.DataFrame(dataset.target, columns=["target"])
    df = pd.concat([data, target], axis=1)
    df.to_csv(
        f"{dataset.DESCR.splitlines()[0].replace(' ', '_').lower().replace('.', '').replace('_', '').replace(':', '')}.csv",
        index=False,
    )
