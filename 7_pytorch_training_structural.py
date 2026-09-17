import pandas as pd
import torch
import matplotlib.pyplot as plt
import torch.nn as nn

#1-Veriyi Hazırla ve İncele (Get & Explore Data)
df = pd.read_csv("06-study_hours_grades.csv")
print(df.head())
print(df.info())

X = torch.tensor(df['study_hours'].values, dtype=torch.float32).unsqueeze(1)
y = torch.tensor(df['grade'].values, dtype=torch.float32).unsqueeze(1)

# we do this because we are going to use nn.Linear which expects float32 rather than other dtype in its weights
# also it expects 2D inputs when using nn.Linear. (batch_size, in_features) and if do not unsquueze it, it won't have that dimension
# just do it without unsquueze and see what happens in the training loop?

print(X.shape)
print(X.ndim)

#2-Train - Test Split
train_split = int(len(X)*0.8)
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

#3-Veriyi Görselleştir (Visualize the Data)
plt.scatter(X_train, y_train, c="b", s=4, label="Training data")
plt.scatter(X_test, y_test, c="g", s=4, label="Testing data")
plt.show()

class LinearRegressionModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.linear_layer = nn.Linear(in_features=1 , out_features=1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)

torch.manual_seed(42)
model = LinearRegressionModel()

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(params = model.parameters(), lr=0.001)

epochs=120

for epoch in range(epochs):
    model.train()
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        test_pred = model(X_test)
        test_loss = loss_fn(test_pred,y_test)

        if epoch % 5 == 0:
            print(f"Epoch: {epoch}, Train Loss: {loss}, Test loss: {test_loss}")

print(model.state_dict)

model.eval()
with torch.inference_mode():
    y_preds = model(X_test)
print(y_preds)
print(y_test)

plt.scatter(X_train, y_train, c="b", s=4, label="Training data")
plt.scatter(X_test, y_test, c="g", s=4, label="Testing data")
plt.scatter(X_test, y_preds, c="r", s=4, label="Predictions")
plt.show()