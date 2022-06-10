import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.datasets as dataset
import torchvision.transforms as transforms
from torchvision import utils

mnist_train = dataset.MNIST(root='MNIST/', train=True, transform=transforms.ToTensor(), download=True)  # Training dataset
mnist_test = dataset.MNIST(root='MNIST/', train=False, transform=transforms.ToTensor(), download=True)  # Test Dataset

train_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=100, shuffle=True, drop_last=True)
x_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float()
y_test = mnist_test.test_labels

# Residual block
class ResidualBlock(torch.nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,kernel_size=3,padding=1):
        super(ResidualBlock,self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(out_channels,out_channels,kernel_size,1,padding),
            torch.nn.BatchNorm2d(out_channels)
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride),
                torch.nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = torch.nn.Sequential()
            
    def forward(self,x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x += self.shortcut(residual)
        x = torch.nn.ReLU(True)(x)
        return x


class CNN(torch.nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.convu1 = torch.nn.Sequential(
        torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
        torch.nn.BatchNorm2d(32),
        torch.nn.ReLU())
    
    self.block1 = torch.nn.Sequential(
            ResidualBlock(32,32))
    
    self.block2 = torch.nn.Sequential(
            ResidualBlock(32,32),
            torch.nn.MaxPool2d(kernel_size=2))
    
    self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(25088, 64),
            torch.nn.ReLU())
    
    self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(64, 10))

    self.logsoftmax = torch.nn.LogSoftmax(dim = 1)
  
  def forward(self, x):
    y = self.convu1(x)
    y = self.block1(y)
    y = self.block1(y)
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

