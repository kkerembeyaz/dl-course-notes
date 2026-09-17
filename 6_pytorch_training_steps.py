#STEPS
    #1-Veriyi Hazırla ve İncele (Get & Explore Data)
    #2-Train - Test Split
    #3-Veriyi Görselleştir (Visualize the Data)
    #4-Modeli Oluştur (Build the Model)
    #5-Eğitim Öncesi (Baseline) Tahmin Yap
    #6-Loss Fonksiyonu ve Optimizer'ı Tanımla
    #7-Training Loop ile Modeli Eğit
    #8-Loss Eğrilerini Görselleştir (Train vs Test)
    #9-Eğitilmiş Modelle Son Tahmini Yap ve Değerlendir


import pandas as pd
import torch
import matplotlib.pyplot as plt

#1-Veriyi Hazırla ve İncele (Get & Explore Data)
df = pd.read_csv("06-study_hours_grades.csv")
print(df.head())
print(df.info())
print(df.describe())

print(type(df["study_hours"].values)) # this is actually a numpy array, so we can convert this easily into tensors
print(torch.tensor(df["study_hours"]))

X = torch.tensor(df['study_hours'].values)
y = torch.tensor(df['grade'].values)

#2-Train - Test Split
train_split = int(len(X)*0.8)
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

#3-Veriyi Görselleştir (Visualize the Data)
plt.scatter(X_train, y_train, c="b", s=4, label="Training data")
plt.scatter(X_test, y_test, c="g", s=4, label="Testing data")
plt.show()

#4-Modeli Oluştur (Build the Model)
from torch import nn

class SimpleLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad = True)
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad = True)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias

torch.manual_seed(42)

model_0 = SimpleLinearRegressionModel()
# model_0 = torch.compile(model_0)

print(list(model_0.parameters()))
print(model_0.state_dict())

#5-Eğitim Öncesi (Baseline) Tahmin Yap
with torch.inference_mode():
    y_pred = model_0(X_test)
    print(y_pred)

plt.scatter(X_train, y_train, c="b", s=4, label="Training data")
plt.scatter(X_test, y_test, c="g", s=4, label="Testing data")
plt.scatter(X_test, y_pred, c="r", s=4, label="Testing data")
plt.show()
#as you can see our prediction is not good at all! because we haven't trained the nn yet

#6-Loss Fonksiyonu ve Optimizer'ı Tanımla
loss_fn = nn.MSELoss()

optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.001)

torch.manual_seed(42)

#7-Training Loop ile Modeli Eğit
epochs = 400
train_loss_values = []
test_loss_values = []
epoch_count = []

for epoch in range(epochs):
    #train mode
    model_0.train()

    y_pred = model_0(X_train)
    loss = loss_fn(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #evaluation mode
    model_0.eval()
    with torch.inference_mode():
        test_pred = model_0(X_test)

        test_loss = loss_fn(test_pred, y_test) # predictions come in torch.float datatype

        if epoch % 5 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(f"Epoch: {epoch} |  Train Loss: {loss} | Test Loss: {test_loss} ")

#8-Loss Eğrilerini Görselleştir (Train vs Test)
plt.plot(epoch_count, train_loss_values, label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.show()

#What did this model come up with?
print(model_0.state_dict())

#9-Eğitilmiş Modelle Son Tahmini Yap ve Değerlendir
model_0.eval()
with torch.inference_mode():
  y_preds = model_0(X_test)

plt.scatter(X_train, y_train, c="b", s=4, label="Training data")
plt.scatter(X_test, y_test, c="g", s=4, label="Testing data")
plt.scatter(X_test, y_preds, c="r", s=4, label="Predictions")
plt.show()