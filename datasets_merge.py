import torch
import numpy as np

class Merge_Datasets(torch.utils.data.Dataset):
  def __init__(self, datasets, get_labels=True):
    self.datasets = datasets
    self.lengths = [len(ds) for ds in datasets]
    self.offsets = np.cumsum(self.lengths)
    self.len = np.sum(self.lengths)
	self.get_labels = get_labels
    
  def __len__(self):
    return self.len
  
  def __getitem__(self, index):
    for dset_num, dset_offset in enumerate(self.offsets):
      if index < dset_offset:
        index -= np.append([0], self.offsets)[dset_num]
        sample, label = self.datasets[dset_num][index]
        domain = torch.tensor(dset_num)
		if self.get_labels:
			domain = label, domain
        return sample, domain