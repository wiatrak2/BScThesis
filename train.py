import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class trainParams:
	def __init__(self):
		pass

def trainWithDomain(**trainParams):
  model_f.train()
  model_c.train()
  model_d.train()

  if train_domain:
    domain_iter = iter(train_loader_domain)
  lambd = -1.
  batch_num = len(train_loader.dataset) / train_loader.batch_size
  for batch_idx, (data, labels) in enumerate(train_loader):
    
    p = ((epoch-1) * batch_num + batch_idx) / (epochs * batch_num)
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
    if extra_loss is not None:
      loss += extra_loss(model_f, model_c, model_d, output, labels)
    train_history['train_loss'].append(loss.item())
    loss.backward()
    
    optim_f.step()
    optim_c.step()
    optim_d.step()
    if train_domain:
      data_snd, _ = domain_iter.next()
      domainData, domains = concatDomainBatches([data.to('cpu'), data_snd.to('cpu')])
      domainData, domains = domainData.to(device), domains.to(device)
      optim_f.zero_grad()
      optim_c.zero_grad()
      optim_d.zero_grad()
      
      if use_lambd:
        lambd = 2. / (1. + np.exp(-10. * p)) - 1.   
      else:
        lambd = 1.
      output = model_d(model_f(domainData), lambd)
      loss_domain = criterion_domain(output, domains)
      if extra_loss is not None:
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
    if batch_idx % log_interval == 0:
        print('Train Epoch: \
              {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, lr: {:.5f} lambd: {:.5f}'
            .format(epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item(), lr, lambd))