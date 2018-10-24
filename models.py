import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GradReverse(torch.autograd.Function):
    def __init__(self, lambd=1.):
      self.lambd = lambd
    
    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)

def grad_reverse(x, lambd):
    return GradReverse(lambd)(x)

class MnistFeatureExtractor(nn.Module):
  def __init__(self):
    super(MnistFeatureExtractor, self).__init__()
    self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    self.conv2_drop = nn.Dropout2d()
  def forward(self, x):
    x = F.leaky_relu(F.max_pool2d(self.conv1(x), 2))
    x = F.leaky_relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
    return x.view(-1, 320)

class MnistClassPredictor(nn.Module):
  def __init__(self, inputSize=320):
    super(MnistClassPredictor, self).__init__()
    self.fc1 = nn.Linear(inputSize, 50)
    self.fc2 = nn.Linear(50, 10)
  def forward(self, x):
    x = F.leaky_relu(self.fc1(x))
    x = F.dropout(x, training=self.training)
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)

class MnistDomain(nn.Module):
  def __init__(self, inputSize=320):
    super(MnistDomain, self).__init__()
    self.fc1 = nn.Linear(inputSize, 100)
    self.fc2 = nn.Linear(100, 2)

  def forward(self, x, lambd=1.):
    x = grad_reverse(x, lambd)
    x = F.leaky_relu(self.fc1(x))
    x = F.dropout(x, training=self.training)
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)

class DomainPredictor(nn.Module):
  def __init__(self, inputSize=320):
    super(DomainPredictor, self).__init__()
    self.fc1 = nn.Linear(inputSize, 100)
    self.fc2 = nn.Linear(100, 2)

  def forward(self, x, *args):
    x = F.leaky_relu(self.fc1(x))
    x = F.dropout(x, training=self.training)
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)

def extendFeatureExtractor(outputSize, inputSize=320):
	return nn.Sequential(MnistFeatureExtractor(), nn.Linear(inputSize, outputSize))

class ZeroHalf(nn.Module):
	def __init__(self, inputSize=320, half=0, useGR=True):
		super(ZeroHalf, self).__init__()
		self.tensor = torch.ones(inputSize).double()
		self.tensor[half*inputSize/2:(half+1)*inputSize/2] = 0
		self.zeroInput = 0

	def forward(self, x, lambd=1.):
		if self.zeroInput:
			x = x * self.tensor
			x = grad_reverse(x, lambd)
		self.zeroInput = 1 - self.zeroInput
		return x.view_as(x)
	