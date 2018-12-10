import torch
import numpy as np

class RandomDataset(torch.utils.data.Dataset):
	def __init__(self, sample_size, samples_num, mean=0., std=1., classes=1, transform=None):
		self.sample_size = sample_size
		self.samples_num = samples_num
		self.mean = mean
		self.std = std
		self.classes = classes
		self.transform = transform
		
	def __len__(self):
		return self.samples_num
	
	def __getitem__(self, index):
		sample = np.random.randn(*self.sample_size).astype(float) * self.std + self.mean
		if self.transform:
			return self.transform(sample).float(), np.random.randint(self.classes)
		return torch.tensor(sample).float(), np.random.randint(self.classes)