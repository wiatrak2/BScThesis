import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GRLog(torch.autograd.Function):
	def __init__(self, lambd=1., grad_log=None):
		self.lambd = lambd
		self.grad_log = grad_log
	
	def forward(self, x):
		return x
	
	def backward(self, grad_output):
		if self.grad_log is not None:
			self.grad_log.append(grad_output)
		return grad_output * -self.lambd
  
class GRandMult(torch.autograd.Function):
	def __init__(self, alpha=1., lambd=1., grad_log=None):
		self.lambd = lambd
		self.alpha = alpha
		self.grad_log = grad_log
	
	def forward(self, x):
		return x
	
	def backward(self, grad_output):
		new_grad = grad_output * torch.rand_like(grad_output) * self.alpha * self.lambd
		if self.grad_log is not None:
			self.grad_log.append(new_grad)
		return new_grad

class GRand(torch.autograd.Function):
	def __init__(self, alpha=1., lambd=1., grad_log=None):
		self.alpha = alpha
		self.lambd = lambd
		self.grad_log = grad_log
	
	def forward(self, x):
		return x
	
	def backward(self, grad_output):
		new_grad = torch.rand_like(grad_output) * self.alpha * self.lambd
		if self.grad_log is not None:
			self.grad_log.append(new_grad)
		return new_grad

class GRInv(torch.autograd.Function):
	def __init__(self, alpha=1., lambd=1., grad_log=None):
		self.lambd = lambd
		self.alpha = alpha
		self.grad_log = grad_log
	
	def forward(self, x):
		return x
	
	def backward(self, grad_output):
		new_grad = (torch.max(torch.abs(grad_output), 1)[0].unsqueeze(1) * torch.sign(grad_output) - grad_output) * self.alpha * self.lambd
		if self.grad_log is not None:
			self.grad_log.append(new_grad)
		return new_grad