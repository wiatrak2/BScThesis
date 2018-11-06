import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SvhnFeatureExtractor(nn.Module):
	def __init__(self, model_size, **kwargs):
		super(SvhnFeatureExtractor, self).__init__()
		self.model_size = model_size
		self.kernel_size = kwargs.get('kernel_size', 5)
		self.padding = kwargs.get('padding', 2)
		self.dropout = kwargs.get('dropout', 0.5)
		self.pool_kernel_size = kwargs.get('pool_kernel_size', 2)
		self.pool_stride = kwargs.get('pool_stride', None)
		self.pool_padding = kwargs.get('pool_padding', 0)
		self.conv_layers = self.get_conv_layers()

	def get_conv_layers(self):
		layers = [nn.Sequential(nn.Conv2d(self.model_size[i], self.model_size[i+1], kernel_size=self.kernel_size, padding=self.padding),
                        nn.BatchNorm2d(self.model_size[i+1]),
                        nn.LeakyReLU(),
                        nn.MaxPool2d(self.pool_kernel_size, stride=self.pool_stride, padding=self.pool_padding),
                        nn.Dropout(self.dropout))
						for i in range(len(self.model_size)-1)]
		return nn.Sequential(*layers)

	def get_output_size(self, device, batched_size=(1,3,32,32)):
		return self.forward(torch.zeros(batched_size).to(device)).size()

	def forward(self, x):
		x = self.conv_layers(x)
		return x.view(x.size(0), -1)	

