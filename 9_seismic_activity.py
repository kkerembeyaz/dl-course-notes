import pandas as pd
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch import nn
import seaborn as sns
from sklearn.model_selection import train_test_split

df = pd.read_csv("08-seismic_activity_svm.csv")
print(df.head)
print(df.describe())
print(df.info())

sns.scatterplot(x=df["underground_wave_energy"],y=df["vibration_axis_variation"],hue=df["seismic_event_detected"])
plt.show()

X = df[["underground_wave_energy" , "vibration_axis_variation"]].values
y = df[["seismic_event_detected"]].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)   # shape: (320, 2)
X_test  = torch.tensor(X_test,  dtype=torch.float32)   # shape: (80, 2)
y_train = torch.tensor(y_train, dtype=torch.float32)  # shape: (320, 1)
y_test  = torch.tensor(y_test,  dtype=torch.float32)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

class ClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(2,16)
        self.layer2 = nn.Linear(16,8)
        self.output = nn.Linear(8,1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.output(x)
        return x
        
model_0 = ClassificationModel()

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params=model_0.parameters(), lr = 0.001)

def calculate_accuracy(y_test,y_pred):
    correct = torch.eq(y_test, y_pred).sum().item()
    accuracy = (correct/len(y_pred))*100
    return accuracy

torch.manual_seed(42)
num_epochs = 300

for epoch in range(num_epochs):

    model_0.train()
    y_logits = model_0(X_train)
    y_pred = torch.round(torch.sigmoid(y_logits))

    loss = loss_fn(y_logits, y_train)
    acc = calculate_accuracy(y_test = y_train, y_pred=y_pred)

    optimizer.zero_grad()          # gradyanları sıfırla
    outputs = model_0(X_train)       # ileri yayılım
    loss = loss_fn(outputs, y_train)
    loss.backward()                # geri yayılım
    optimizer.step()               # ağırlıkları güncelle

    model_0.eval()
    with torch.inference_mode():
        test_logits = model_0(X_test)
        test_pred = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_fn(test_logits,y_test)
        test_acc = calculate_accuracy(y_test=y_test, y_pred=test_pred)

        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")