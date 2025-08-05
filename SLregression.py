import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
try:
    df = pd.read_csv(input("dataset: "))
except FileNotFoundError:
    print("Error: no such dataset")
    exit()

X = df.iloc[:, 0].values.reshape(-1, 1)  # height
y = df.iloc[:, 1].values.reshape(-1, 1)  # weight

# to torch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# normalising for 
X_mean, X_std = X_train.mean(), X_train.std()
y_mean, y_std = y_train.mean(), y_train.std()

X_train = (X_train- X_mean) / X_std #actual -mean) /std
y_train = (y_train - y_mean) / y_std

X_test_norm = (X_test - X_mean) / X_std
y_test_norm = (y_test - y_mean) / y_std

# Model Class
class SLR(nn.Module):
    def __init__(self):
        super(SLR, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = SLR()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0005)

# Training loop
epochs = 4000
for epoch in range(epochs):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 500 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Getting learned parameters 
[w, b] = model.parameters()
print(f"\nLearned weight: {w.item():.4f}, bias: {b.item():.4f}")

# prdict n denormalise
with torch.no_grad():
    y_pred_norm = model(X_test_norm)
    y_pred = y_pred_norm * y_std + y_mean  # denormalsing the prediction 
    y_actual = y_test  # already in original scale
    X_actual = X_test

#plot
plt.scatter(X_actual, y_actual, color="blue", label="Actual")
plt.scatter(X_actual, y_pred, color="red", label="Predicted")
plt.xlabel("Height")
plt.ylabel("Weight")
plt.title("Simple Linear Regression")
plt.legend()
plt.grid(True)
plt.show()
