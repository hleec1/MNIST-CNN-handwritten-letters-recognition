import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.datasets as dataset
import torchvision.transforms as transforms
from torchvision import utils

# Downloading the MNIST dataset
mnist_train = dataset.MNIST(root='MNIST/', train=True, transform=transforms.ToTensor(), download=True)  # Training dataset
mnist_test = dataset.MNIST(root='MNIST/', train=False, transform=transforms.ToTensor(), download=True)  # Test Dataset

train_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=100, shuffle=True, drop_last=True)
x_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float()
y_test = mnist_test.test_labels

class CNN(torch.nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.convu1 = torch.nn.Sequential(
        torch.nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU())
           
    self.convu2 = torch.nn.Sequential(
          torch.nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
          torch.nn.ReLU(),
          torch.nn.MaxPool2d(kernel_size=2))
    
    self.convu3 = torch.nn.Sequential(
          torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
          torch.nn.ReLU())
    
    self.convu4 = torch.nn.Sequential(
          torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
          torch.nn.ReLU(),
          torch.nn.MaxPool2d(kernel_size=2))

    self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(3136, 64),
            torch.nn.ReLU())
    
    self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(64, 10))

    self.logsoftmax = torch.nn.LogSoftmax(dim = 1)
  
  def forward(self, x):
    y = self.convu1(x)
    y = self.convu2(y)
    y = self.convu3(y)
    y = self.convu4(y)
    y = y.view(y.size(0), -1) 
    y = self.fc1(y)
    y = self.fc2(y)
    y = self.logsoftmax(y)
    return y
    
epochs = 30
batch_size = train_loader.batch_size
learning_rate = 0.001
len_trainloader = len(train_loader)
model = CNN()
loss_f = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
train_loss = []
test_acc = []
for epoch in range(epochs):
  avg_loss = 0
  items = 0
  for x, y in train_loader: 
    optimizer.zero_grad()
    hypothesis = model(x)
    loss = loss_f(hypothesis, y)
    loss.backward()
    optimizer.step()
    avg_loss += loss / len_trainloader
    items += loss.item() / len_trainloader
  with torch.no_grad():
    prediction = model(x_test)
    correct_prediction = torch.argmax(prediction, 1) == y_test
    accuracy = correct_prediction.float().mean()
    test_acc.append(accuracy.item())
  train_loss.append(items)
  print(f'epoch: {epoch+1}, loss: {avg_loss}, test_acc: {accuracy}')
print(f'Final Loss: {avg_loss}')
print(f'Final Test Acc: {accuracy}')
