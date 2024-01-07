import torch 
import torch.nn as nn
import torch.optim as optim
from utils import SimpleCNN, CustomDataset
from torch.utils.data import  DataLoader
from torchvision import transforms
from torchmetrics.classification import MulticlassAccuracy

n_epochs = 10
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01


transform = transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])

custom_dataset = CustomDataset(csv_file='./wrong_result/predictions.csv', root_dir='./wrong_result/images/', transform=transform)
data_loader = DataLoader(custom_dataset, batch_size=batch_size_train, shuffle=True)
device = "cuda" if torch.cuda.is_available else "cpu"

model = SimpleCNN()
model.load_state_dict(torch.load('model.pth'))
model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = learning_rate)
metric = MulticlassAccuracy(num_classes=10).to(device)

for epoch in range(n_epochs):
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{n_epochs}, Loss: {loss.item()}')

model.eval()

for data, target in data_loader :
  train_acc = 0
  data = data.to(device)
  target = target.to(device)
  with torch.no_grad():
    y_pred = model(data)
    loss = loss_fn(y_pred, target)
    train_acc = metric(y_pred, target)
    print(train_acc)

torch.save(model.state_dict(), 'model.pth')

