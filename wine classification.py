import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report


# -------------------------------
# 1. Load CSV Dataset
# -------------------------------
df = pd.read_csv("winequality-red.csv")

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nColumns:")
print(df.columns)

print("\nMissing values:")
print(df.isnull().sum())


# -------------------------------
# 2. Separate Features and Target
# -------------------------------
target_column = "quality"

X = df.drop(columns=[target_column])
y = df[target_column]

print("\nFeature shape:", X.shape)
print("Target shape:", y.shape)

print("\nQuality classes before encoding:")
print(sorted(y.unique()))


# -------------------------------
# 3. Encode Target Labels
# -------------------------------
# Original quality labels are 3, 4, 5, 6, 7, 8
# PyTorch CrossEntropyLoss needs labels like 0, 1, 2, 3, 4, 5

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print("\nQuality classes after encoding:")
print(np.unique(y))

num_classes = len(np.unique(y))
input_size = X.shape[1]

print("\nNumber of input features:", input_size)
print("Number of output classes:", num_classes)


# -------------------------------
# 4. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# -------------------------------
# 5. Feature Scaling
# -------------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# -------------------------------
# 6. Convert Data into PyTorch Tensors
# -------------------------------
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)


# -------------------------------
# 7. Create DataLoader
# -------------------------------
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=32,
    shuffle=False
)


# -------------------------------
# 8. Define Neural Network Model
# -------------------------------
class WineClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(WineClassifier, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.network(x)


model = WineClassifier(input_size=input_size, num_classes=num_classes)

print("\nModel Architecture:")
print(model)


# -------------------------------
# 9. Loss Function and Optimizer
# -------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# -------------------------------
# 10. Training Loop
# -------------------------------
epochs = 100

for epoch in range(epochs):
    model.train()

    total_loss = 0

    for features, labels in train_loader:
        # Forward pass
        outputs = model(features)

        # Loss calculation
        loss = criterion(outputs, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {average_loss:.4f}")


# -------------------------------
# 11. Model Evaluation
# -------------------------------
model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():
    for features, labels in test_loader:
        outputs = model(features)

        _, predicted = torch.max(outputs, 1)

        all_predictions.extend(predicted.numpy())
        all_labels.extend(labels.numpy())


# -------------------------------
# 12. Accuracy
# -------------------------------
accuracy = accuracy_score(all_labels, all_predictions)

print("\nTest Accuracy:")
print(f"{accuracy * 100:.2f}%")


# -------------------------------
# 13. Classification Report
# -------------------------------
original_class_names = [str(cls) for cls in label_encoder.classes_]

print("\nClassification Report:")
print(classification_report(
    all_labels,
    all_predictions,
    target_names=original_class_names,
    zero_division=0
))