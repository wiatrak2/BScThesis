import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict, namedtuple
from copy import deepcopy
import test_model

class Trainer:

	def __init__(self, models, optims, criterions, device, **kwargs):
		self.models = models
		self.optims = optims
		self.criterions = criterions
		self.device = device
		self.use_lambd = kwargs.get('use_lambd', True)
		self.default_lambd = kwargs.get('default_lambd', 1.0)
		self.tune_lr = kwargs.get('tune_lr', False)
		self.train_history = kwargs.get('train_history', defaultdict(lambda:[]))
		self.train_domain = kwargs.get('train_domain', True)
		self.log_interval = kwargs.get('log_interval', 100)
		self.print_logs = kwargs.get('print_logs', True)
		self.best_accuracy = 0.0
		self.best_model = None

	@staticmethod
	def concat_domain_batches(batches, shuffle=True):
		domain_num = np.arange(len(batches))[...,None]
		min_batch_size = min([len(batch) for batch in batches])
		batches = [batch[:min_batch_size] for batch in batches]
		batch_len = min_batch_size
		try:
			batches = torch.cat(batches).numpy()
		except:
			pass 
		domain_labels = (np.ones(batch_len) * domain_num).reshape(-1)
		if shuffle:
			idx = np.random.permutation(len(domain_labels))
			batches, domain_labels = batches[idx], domain_labels[idx].astype(np.long)
		return torch.from_numpy(batches), torch.from_numpy(domain_labels)

	def _train_with_domain(self, loaders, epoch):
		model_f = self.models.model_f.train()
		model_c = self.models.model_c.train()
		model_d = self.models.model_d.train()

		optim_f = self.optims.optim_f
		optim_c = self.optims.optim_c
		optim_d = self.optims.optim_d
		
		train_loader = loaders.train_loader
		criterion = self.criterions.criterion

		if self.train_domain:
			domain_iter = iter(loaders.train_loader_domain)
			criterion_domain = self.criterions.criterion_domain

		batch_num = len(train_loader.dataset) / train_loader.batch_size
		lambd = self.default_lambd

		for batch_idx, (data, labels) in enumerate(train_loader):
			
			p = ((epoch-1) * batch_num + batch_idx) / (self.epochs * batch_num)
			if self.tune_lr:
				lr = 0.01 / (1. + 10. * p)**0.75
				optim_f.lr = lr
				optim_c.lr = lr
				optim_d.lr = lr
			
			data = data.to(self.device)
			labels = labels.to(self.device)
			optim_f.zero_grad()
			optim_c.zero_grad()
			optim_d.zero_grad()
			output = model_c(model_f(data))
			loss = criterion(output, labels)
			if self.extra_loss is not None:
				loss += self.extra_loss(model_f, model_c, model_d, output, labels)
			self.train_history['train_loss'].append(loss.item())
			loss.backward()
			
			optim_f.step()
			optim_c.step()
			optim_d.step()
			if self.train_domain:
				try:
					data_snd, _ = domain_iter.next()
				except StopIteration:
					break
				domainData, domains = self.concat_domain_batches([data.to('cpu'), data_snd.to('cpu')])
				domainData, domains = domainData.to(self.device), domains.to(self.device)
				optim_f.zero_grad()
				optim_c.zero_grad()
				optim_d.zero_grad()
				
				if self.use_lambd:
					lambd = 2. / (1. + np.exp(-10. * p)) - 1.   
				else:
					lambd = self.default_lambd

				output = model_d(model_f(domainData), lambd)
				loss_domain = criterion_domain(output, domains)
				if self.extra_loss is not None:
					loss_domain += self.extra_loss(model_f, model_c, model_d, output, labels)
				self.train_history['domain_loss'].append(loss_domain.item())
				
				loss_domain.backward()
				optim_f.step()
				optim_c.step()
				optim_d.step()
				
				model_d_mtx = model_d.get_mtx().weight.cpu().detach().numpy()
				model_c_mtx = model_c.get_mtx().weight.cpu().detach().numpy()
				self.train_history['avg_len_c'].append(np.mean(np.diag(model_c_mtx.dot(model_c_mtx.T))))
				self.train_history['avg_len_d'].append(np.mean(np.diag(model_d_mtx.dot(model_d_mtx.T))))
				self.train_history['avg_dot'].append(np.mean(np.power(model_c_mtx.dot(model_d_mtx.T), 2)))  
			if batch_idx % self.log_interval == 0 and self.print_logs:
				print('Train Epoch: \
					{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, lr: {:.5f} lambd: {:.5f}'
					.format(epoch, batch_idx * len(data), len(train_loader.dataset),
					100. * batch_idx / len(train_loader), loss.item(), lr if self.tune_lr else 0., lambd))

	def _test_domain_model(self, loaders, test_history):
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
			source_test_loss, source_correct = test_model.test_model(class_model, self.device,
													self.criterions, source_test_loader, no_print=True)
			target_test_loss, target_correct = test_model.test_model(class_model, self.device,
													self.criterions, target_test_loader, no_print=True)
			
			for data, target in merged_test_loader:
				data = data.to(self.device)
				if merged_test_loader.dataset.get_labels:
					_, domains = target
				else:
					domains = target
				domains = domains.to(self.device)
				
				domain_out = domain_model(data)
				domain_pred = domain_out.max(1, keepdim=True)[1] 
				domain_correct += domain_pred.eq(domains.view_as(domain_pred)).sum().item()
			
		domain_test_loss /= len(merged_test_loader.dataset)
		
		test_history['target_loss'].append(target_test_loss)
		test_history['source_loss'].append(source_test_loss)
		test_history['target_acc'].append(100. * target_correct / len(target_test_loader.dataset))
		test_history['source_acc'].append(100. * source_correct / len(source_test_loader.dataset))
		test_history['domain_acc'].append(100. * domain_correct / len(merged_test_loader.dataset))
		if self.print_logs:
			print('\nTarget Domain Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
				target_test_loss, target_correct, len(target_test_loader.dataset),
				100. * target_correct / len(target_test_loader.dataset)))
			print('Source Domain Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
				source_test_loss, source_correct, len(source_test_loader.dataset),
				100. * source_correct / len(source_test_loader.dataset)))
			print('Domains predictor:  Accuracy: {}/{} ({:.0f}%)\n'.format(
				domain_correct, len(merged_test_loader.dataset),
				100. * domain_correct / len(merged_test_loader.dataset)))
		return 100. * target_correct / len(target_test_loader.dataset)
	
	def train(self, epochs, loaders, extra_loss=None, test_history=None):
		self.epochs = epochs
		self.extra_loss = extra_loss
		if test_history is None:
			test_history = defaultdict(lambda:[])
		for epoch in range(1, self.epochs+1):
			self._train_with_domain(loaders, epoch)
			acc = self._test_domain_model(loaders, test_history)
			if acc > self.best_accuracy:
				self.best_model = deepcopy(self.models)
				self.best_accuracy = acc
	
	def get_best_model(self):
		return self.best_model, self.best_accuracy