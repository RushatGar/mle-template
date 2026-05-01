
#!pip install torch torchvision numpy sklearn matplotlib
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA

# Загружаем train, valid, test с target_type='attr' (40 признаков)
train_dataset = CelebA(root='./data', split='train', target_type='attr', download=True)
valid_dataset = CelebA(root='./data', split='valid', target_type='attr')
test_dataset = CelebA(root='./data', split='test', target_type='attr')
print(f"Train: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}")
# Преобразуем в тензоры признаков и метки нужного атрибута
def get_XY(dataset, attr_index=20): # 20 → Male (индекс из 40)
    X = []
    y = []
    for i in range(len(dataset)):
        attrs = dataset[i][1]  # тензор (40,)
        X.append(attrs.float())
        y.append(attrs[attr_index].float())
    return torch.stack(X), torch.tensor(y).reshape(-1,1)
X_train, y_train = get_XY(train_dataset)
X_valid, y_valid = get_XY(valid_dataset)
X_test, y_test = get_XY(test_dataset)

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(self.linear(x))
model = LogisticRegressionModel(input_dim=40)
criterion = nn.BCELoss() # Бинарная кросс-энтропия
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
batch_size = 64
train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True)

epochs = 5
model.train()
for epoch in range(epochs):
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X.float())
        loss = criterion(outputs, batch_y.float().view(-1, 1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}')

model.eval()
with torch.no_grad():
    test_pred = model(X_test.float())
    test_pred_labels = (test_pred >= 0.5).float().numpy()
    test_true = y_test.numpy()
    acc = accuracy_score(test_true, test_pred_labels)
    print(f'Test Accuracy: {acc:.4f}')
    print('Confusion Matrix:')
    print(confusion_matrix(test_true, test_pred_labels))
    print(classification_report(test_true, test_pred_labels, target_names=['Female', 'Male']))

coeffs = model.linear.weight.detach().numpy().flatten()
features = list(train_dataset.attr_names)[:40]  # список из 40 атрибутов
top_positive = sorted(zip(features, coeffs), key=lambda x: x[1], reverse=True)[:5]
top_negative = sorted(zip(features, coeffs), key=lambda x: x[1])[:5]
print("Top-5 положительных весов (Male=1):", top_positive)
print("Top-5 отрицательных весов (Female=1):", top_negative)
plt.figure(figsize=(10, 6))
plt.barh(features, coeffs, color='steelblue')
plt.xlabel('Вес')
plt.title('Коэффициенты логистической регрессии для предсказания Male')
plt.grid(axis='x')
plt.show()

# после того как модель обучена
torch.save(model.state_dict(), 'model.pkl')
# также сохраним мета-информацию (размер входа)
meta = {'input_dim': 40, 'attr_names': train_dataset.attr_names[:40]}
torch.save(meta, 'meta.pkl')