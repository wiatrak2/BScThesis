import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def test(model, device, test_loader, no_print=False):
	model.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			output = model(data)
			test_loss += criterion(output, target, reduction='sum').item()
			pred = output.max(1, keepdim=True)[1] 
			correct += pred.eq(target.view_as(pred)).sum().item()

	test_loss /= len(test_loader.dataset)
	if no_print:
		return test_loss, correct
	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset),
		100. * correct / len(test_loader.dataset)))