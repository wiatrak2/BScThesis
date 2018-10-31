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
        return grad_output * -self.lambd

def grad_reverse(x, lambd):
    return GradReverse(lambd)(x)

class MnistFeatureExtractor(nn.Module):
	def __init__(self):
		super(MnistFeatureExtractor, self).__init__()
		self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
		self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
		self.conv2_drop = nn.Dropout2d()

	def get_mtx(self):
		return None

	def forward(self, x):
		x = F.leaky_relu(F.max_pool2d(self.conv1(x), 2))
		x = F.leaky_relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
		return x.view(x.size(0), -1)


class MnistClassPredictor(nn.Module):
	def __init__(self, input_size=320, inner_size=100):
		super(MnistClassPredictor, self).__init__()
		self.fc1 = nn.Linear(input_size, inner_size)
		self.fc2 = nn.Linear(inner_size, 10)

	def get_mtx(self):
		return self.fc1

	def forward(self, x):
		x = F.leaky_relu(self.fc1(x))
		x = F.dropout(x, training=self.training)
		x = self.fc2(x)
		return F.log_softmax(x, dim=1)

class MnistDomain(nn.Module):
	def __init__(self, input_size=320, inner_size=100):
		super(MnistDomain, self).__init__()
		self.fc1 = nn.Linear(input_size, inner_size)
		self.fc2 = nn.Linear(inner_size, 2)

	def get_mtx(self):
		return self.fc1

	def forward(self, x, lambd=1.):
		x = grad_reverse(x, lambd)
		x = F.leaky_relu(self.fc1(x))
		x = F.dropout(x, training=self.training)
		x = self.fc2(x)
		return F.log_softmax(x, dim=1)

class DomainPredictor(nn.Module):
	def __init__(self, input_size=320, inner_size=100):
		super(DomainPredictor, self).__init__()
		self.fc1 = nn.Linear(input_size, inner_size)
		self.fc2 = nn.Linear(inner_size, 2)

	def get_mtx(self):
		return self.fc1

	def forward(self, x, *args):
		x = F.leaky_relu(self.fc1(x))
		x = F.dropout(x, training=self.training)
		x = self.fc2(x)
		return F.log_softmax(x, dim=1)

class CutHalf(nn.Module):
	def __init__(self, output_model, half=0):
		super(CutHalf, self).__init__()
		self.half = half
		self.output_model = output_model
		
	def get_mtx(self):
		return self.output_model.get_mtx()
	
	def forward(self, x, *args):
		x = x[:, :x.size()[1]//2] if self.half == 0 else x[:, x.size()[1]//2:]
		if args:
			return self.output_model(x, *args)
		return self.output_model(x)

class LinearFromList(nn.Module):
	def __init__(self, size_list, use_gr=False, output_model=None):
		super(LinearFromList, self).__init__()
		self.size_list = size_list
		self.linears = nn.ModuleList([nn.Linear(size_list[i], size_list[i+1]) for i in range(len(size_list)-1)])
		self.output_model = output_model
		self.use_gr = use_gr

	def get_mtx(self):
		return self.linears[0] if len(self.linears) > 0 else self.output_model.get_mtx()

	def split(self, split_after):
		if split_after > len(self.linears):
			return self, None
		new_model = LinearFromList(self.size_list[:split_after+1], use_gr=self.use_gr)
		new_model_out = LinearFromList(self.size_list[split_after:], use_gr=self.use_gr, output_model=self.output_model)
		return new_model, new_model_out

	def forward(self, x, lambd=1.):
		if self.use_gr:
			x = grad_reverse(x, lambd)
		for layer in self.linears:
			x = F.leaky_relu(layer(x))
			x = F.dropout(x, training=self.training)
		if self.output_model is not None:
			return self.output_model(x)
		return x

def extend_feature_extractor(model_f, model_continuation, freeze_model=True, return_size=True):
	if freeze_model:
		for param in model_f.parameters():
			param.requires_grad = False
	new_model_f = nn.Sequential(model_f, model_continuation)
	if return_size:
		return new_model_f, model_continuation.get_mtx().out_features
	return new_model_f

def get_models(model_f_linear, model_c_linear, model_d_linear, use_gr=False, model_f_dropout=False):
	model_f = nn.Sequential(MnistFeatureExtractor(), LinearFromList(model_f_linear))
	if not model_f_dropout:
		model_f = nn.Sequential(model_f, nn.Linear(model_f_linear[-1], model_f_linear[-1]))
	model_c = LinearFromList(model_c_linear, output_model=nn.Sequential(nn.Linear(model_c_linear[-1], 10), nn.LogSoftmax(dim=1)))
	model_d = LinearFromList(model_d_linear[:-1], use_gr, output_model=DomainPredictor(model_d_linear[-2], model_d_linear[-1]))
	return model_f, model_c, model_d