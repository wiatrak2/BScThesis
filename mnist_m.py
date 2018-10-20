import pickle
import torch
import torchvision
import numpy as np

class Mnist_M(torch.utils.data.Dataset):
  
  def __init__(self, dataset_path, train=True, transform=None):
    
    self.train = train
    self.transform = transform
    
    with open(dataset_path, 'rb') as mnist_m:
      mnist_m_data = pickle.load(mnist_m, encoding='bytes')
    mnist_m_train_data = torch.ByteTensor(mnist_m_data[b'train'])
    mnist_m_test_data = torch.ByteTensor(mnist_m_data[b'test'])
    if train:
      mnist_m_train_labels = torchvision.datasets.MNIST(
          root='./data', train=True, download=True).train_labels
      self.mnist_m_set = list(zip(mnist_m_train_data, mnist_m_train_labels))
    else:
      mnist_m_test_labels = torchvision.datasets.MNIST(
          root='./data', train=False, download=True).test_labels
      self.mnist_m_set = list(zip(mnist_m_test_data, mnist_m_test_labels))
    self.len = len(self.mnist_m_set)
  
  def __len__(self):
    return self.len
  
  def __getitem__(self, index):
    label = self.mnist_m_set[index][1]
    sample = self.mnist_m_set[index][0].permute(2,0,1).float()
    sample = sample / 255
    
    if self.transform: 
      self.transform(sample)
      
    return (sample, label)