import pathlib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import KFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder

import numpy as np
import pandas as pd


def check_device():
    print(f"PyTorch Version: {torch.__version__}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    return device


# Class Weight Calculation
def compute_class_weights(y):
    y_np = y.numpy().astype(int)
    classes = np.unique(y_np)
    class_weights = compute_class_weight("balanced", classes=classes, y=y_np)
    return torch.tensor(class_weights, dtype=torch.float32)


# Model Definition
class SimpleNN(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=64):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)  # Output raw logits


def load_data(file_path):
    df = pd.read_csv(file_path)
    feature_cols = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
    target_col = "Species"

    X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    y = LabelEncoder().fit_transform(df[target_col].values)
    y = torch.tensor(y, dtype=torch.long)

    return X, y


# Training Loop with Early Stopping
def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    patience=5,
    num_epochs=50,
):
    best_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(
            f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        # Early Stopping Check
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break


def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    test_loss /= len(test_loader)
    accuracy = correct / total
    print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")
    return test_loss, accuracy


def kfold_validation(
    X, y, input_dim, num_classes, device, n_splits=5, batch_size=32, lr=0.001
):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\nFold {fold + 1}/{n_splits}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_data = TensorDataset(X_train, y_train)
        val_data = TensorDataset(X_val, y_val)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size)

        model = SimpleNN(input_dim, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        train_model(model, train_loader, val_loader, criterion, optimizer, device)

        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
        fold_metrics.append((val_loss, val_accuracy))

    avg_loss = np.mean([x[0] for x in fold_metrics])
    avg_accuracy = np.mean([x[1] for x in fold_metrics])
    print(f"\nAverage Loss: {avg_loss:.4f}, Average Accuracy: {avg_accuracy:.4f}")
    return fold_metrics


def main():
    device = check_device()
    dataset_path = (
        pathlib.Path(__file__)
        .resolve()
        .parent.parent.parent.joinpath("data", "iris.csv")
    )

    X, y = load_data(dataset_path)
    input_dim = X.shape[1]
    num_classes = len(torch.unique(y))

    kfold_validation(X, y, input_dim, num_classes, device)


if __name__ == "__main__":
    main()
