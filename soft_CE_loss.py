import torch
from torch.nn import CrossEntropyLoss

def soft_cross_entropy(pred, targets, crit=CrossEntropyLoss()):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	loss = 0
	w = len(targets)
	for t in targets:
		t = t.to(device)
		loss += (1.0/w) * crit(pred, t)
	return loss