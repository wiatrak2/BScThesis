import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple
from test import test

class Trainer:

	def __init__(self, models, optims, criterions, device, **kwargs):
		self.models = models
		self.optims = optims
		self.criterions = criterions
		self.device = device
		self.use_lambd = kwargs.get('use_lambd', True)
		self.tune_lr = kwargs.get('tune_lr', False)
		self.train_domain = kwargs.get('train_domain', True)
		self.log_interval = kwargs.get('log_interval', 100)

	@staticmethod
	def concat_domain_batches(batches, shuffle=True):
		domain_num = np.arange(len(batches))[...,None]
		batch_len = len(batches[0])
		try:
			batches = torch.cat(batches).numpy()
		except:
			pass 
		domain_labels = (np.ones(batch_len) * domain_num).reshape(-1)
		if shuffle:
			idx = np.random.permutation(len(domain_labels))
			batches, domain_labels = batches[idx], domain_labels[idx].astype(np.long)
		return torch.from_numpy(batches), torch.from_numpy(domain_labels)

	def _train_with_domain(self, loaders, epoch, train_history=defaultdict(lambda:[])):
		model_f = self.models.model_f.train()
		model_c = self.models.model_c.train()
		model_d = self.models.model_d.train()
		train_loader = loaders.train_loader
		criterion = self.criterions.criterion

		if self.train_domain:
			domain_iter = iter(loaders.train_loader_domain)
			criterion_domain = self.criterions.criterion_domain

		lambd = -1.
		batch_num = len(train_loader.dataset) / train_loader.batch_size

		for batch_idx, (data, labels) in enumerate(train_loader):
			
			p = ((epoch-1) * batch_num + batch_idx) / (epochs * batch_num)
			if self.tune_lr:
				lr = 0.01 / (1. + 10. * p)**0.75
				optim_f.lr = lr
				optim_c.lr = lr
				optim_d.lr = lr
			
			data = data.to(device)
			labels = labels.to(device)
			optim_f.zero_grad()
			optim_c.zero_grad()
			optim_d.zero_grad()
			output = model_c(model_f(data))
			loss = criterion(output, labels)
			if self.extra_loss is not None:
				loss += extra_loss(model_f, model_c, model_d, output, labels)
			train_history['train_loss'].append(loss.item())
			loss.backward()
			
			optim_f.step()
			optim_c.step()
			optim_d.step()
			if train_domain:
				data_snd, _ = domain_iter.next()
				domainData, domains = self.concat_domain_batches([data.to('cpu'), data_snd.to('cpu')])
				domainData, domains = domainData.to(device), domains.to(device)
				optim_f.zero_grad()
				optim_c.zero_grad()
				optim_d.zero_grad()
				
				if self.use_lambd:
					lambd = 2. / (1. + np.exp(-10. * p)) - 1.   
				else:
					lambd = 1.

				output = model_d(model_f(domainData), lambd)
				loss_domain = criterion_domain(output, domains)
				if self.extra_loss is not None:
					loss_domain += extra_loss(model_f, model_c, model_d, output, labels)
				train_history['domain_loss'].append(loss_domain)
				
				loss_domain.backward()
				optim_f.step()
				optim_c.step()
				optim_d.step()
				
				model_d_mtx = model_d.fc1.weight.cpu().detach().numpy()
				model_c_mtx = model_c.fc1.weight.cpu().detach().numpy()
				train_history['avg_len_c'].append(np.mean(np.diag(model_c_mtx.dot(model_c_mtx.T))))
				train_history['avg_len_d'].append(np.mean(np.diag(model_d_mtx.dot(model_d_mtx.T))))
				train_history['avg_dot'].append(np.mean(model_c_mtx.dot(model_d_mtx.T)))  
			if batch_idx % self.log_interval == 0:
				print('Train Epoch: \
					{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, p: {:.5f} lambd: {:.5f}'
					.format(epoch, batch_idx * len(data), len(train_loader.dataset),
					100. * batch_idx / len(train_loader), loss.item(), p, lambd))

	def _test_domain_model(loaders, test_history=defaultdict(lambda:[])):
		model_f = self.models.model_f.eval()
		model_c = self.models.model_c.eval()
		model_d = self.models.model_d.eval()
		source_test_loader = loaders.source_test_loader
		target_test_loader = loaders.target_test_loader
        merged_test_loader = loaders.merged_test_loader

		domain_test_loss = 0
		domain_correct = 0
		
		with torch.no_grad():
			class_model = nn.Sequential(model_f, model_c)
			domain_model = nn.Sequential(model_f, model_d)
			source_test_loss, source_correct = test(class_model, device,
													source_test_loader, no_print=True)
			target_test_loss, target_correct = test(class_model, device,
													target_test_loader, no_print=True)
			
			for data, target in merged_test_loader:
			data = data.to(device)
			_, domains = target
			domains = domains.to(device)
			
			domain_out = domain_model(data)
			domain_pred = domain_out.max(1, keepdim=True)[1] 
			domain_correct += domain_pred.eq(domains.view_as(domain_pred)).sum().item()
			
		domain_test_loss /= len(merged_test_loader.dataset)
		
		test_history['target_loss'].append(target_test_loss)
		test_history['source_loss'].append(source_test_loss)
		test_history['target_acc'].append(100. * target_correct / len(target_test_loader.dataset))
		test_history['source_acc'].append(100. * source_correct / len(source_test_loader.dataset))
		test_history['domain_acc'].append(100. * domain_correct / len(merged_test_loader.dataset))
		
		print('\nTarget Domain Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
			target_test_loss, target_correct, len(target_test_loader.dataset),
			100. * target_correct / len(target_test_loader.dataset)))
		print('\nSource Domain Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
			source_test_loss, source_correct, len(source_test_loader.dataset),
			100. * source_correct / len(source_test_loader.dataset)))
		print('\nDomains predictor:  Accuracy: {}/{} ({:.0f}%)\n'.format(
			domain_correct, len(merged_test_loader.dataset),
			100. * domain_correct / len(merged_test_loader.dataset)))
	
	def train(epochs, loaders, test_history = defaultdict(lambda:[]), train_history = defaultdict(lambda:[])):
		self.epochs = epochs
		for epoch in range(1, self.epochs+1):
			_train_with_domain(loaders, epoch, train_history)
			_test_domain_model(loaders, test_history)