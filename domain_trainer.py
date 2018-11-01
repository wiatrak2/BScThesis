import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict, namedtuple

class DomainTrainer:

	def __init__(self, models, optims, criterions, device, **kwargs):
		self.models = models
		self.optims = optims
		self.criterions = criterions
		self.device = device
		self.history = kwargs.get('history', True)
		self.log_interval = kwargs.get('log_interval', 100)
		self.print_logs = kwargs.get('print_logs', True)

	def _train_domain(self, loaders, gr_models, epoch, train_history):
		model_d = self.models.model_d.train()
		model_f = self.models.model_f.eval()
		train_loader = loaders.merged_test_loader
		optimizer = self.optims.optim_d
		criterion_domain = self.criterions.criterion_domain

		if gr_models is not None:
			model_c = gr_models.model_c
			model_gr = gr_models.model_d

		for batch_idx, (data, (_, domains)) in enumerate(train_loader):
			data, domains = data.to(self.device), domains.to(self.device)
			optimizer.zero_grad()
			output = model_d(model_f(data))
			loss = criterion_domain(output, domains)
			loss.backward()
			optimizer.step()
			if self.history and gr_models:
				model_c_mtx = model_c.fc1.weight.cpu().detach().numpy()
				model_d_mtx = model_d.fc1.weight.cpu().detach().numpy()
				model_gr_mtx = model_gr.fc1.weight.cpu().detach().numpy()
				train_history['avg_len'].append(np.mean(np.diag(model_d_mtx.dot(model_d_mtx.T))))
				train_history['avg_dot'].append(np.mean(model_d_mtx.dot(model_c_mtx.T)))
				train_history['avg_dot_gr'].append(np.mean(model_d_mtx.dot(model_gr_mtx.T)))
			if batch_idx % self.log_interval == 0 and self.print_logs:
				print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
					epoch, batch_idx * len(data), len(train_loader.dataset),
					100. * batch_idx / len(train_loader), loss.item()))
		
	@staticmethod
	def test_domain_pred(model, device, merged_test_loader, print_logs=True):
		model.eval()
				
		domain_test_loss = 0
		domain_correct = 0
				
		with torch.no_grad():    
			for data, target in merged_test_loader:
				data = data.to(device)
				_, domains = target
				domains = domains.to(device)
					
				domain_out = model(data)
				domain_pred = domain_out.max(1, keepdim=True)[1] 
				domain_correct += domain_pred.eq(domains.view_as(domain_pred)).sum().item()
				
		domain_test_loss /= len(merged_test_loader.dataset)
		if print_logs:
			print('\nDomains predictor:  Accuracy: {}/{} ({:.0f}%)\n'.format(
				domain_correct, len(merged_test_loader.dataset),
				100. * domain_correct / len(merged_test_loader.dataset)))	
					
	def train(self, epochs, loaders, gr_models=None, train_history=None):
		self.epochs = epochs
		if train_history is None:
			train_history = defaultdict(lambda:[])
		for epoch in range(1, self.epochs+1):
			self._train_domain(loaders, gr_models, epoch, train_history)
			domain_model = nn.Sequential(self.models.model_f, self.models.model_d)
			self.test_domain_pred(domain_model, self.device, loaders.merged_test_loader, print_logs=self.print_logs)			