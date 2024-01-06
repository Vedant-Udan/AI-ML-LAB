import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
  def __init__(self):
    super(SimpleCNN, self).__init__()
    self.Conv1 = nn.Conv2d(1, 10, kernel_size = 5 )
    self.Conv2 = nn.Conv2d(10, 20, kernel_size = 3)
    self.Conv2_drop = nn.Dropout2d()
    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(500, 100)
    self.fc2 = nn.Linear(100, 10)

  def forward(self, x):
    x = F.relu(F.max_pool2d(self.Conv1(x), 2))
    x = F.relu(F.max_pool2d(self.Conv2_drop(self.Conv2(x)), 2))
    x = self.flatten(x)
    x = F.relu(self.fc1(x))
    x = F.log_softmax(self.fc2(x), dim=1)
    return x