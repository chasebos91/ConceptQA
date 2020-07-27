import torch
from torch.nn import CrossEntropyLoss

def soft_cross_entropy(pred, targets, crit=CrossEntropyLoss()):
	loss = 0
	w = len(targets)
	for t in targets:
		loss += (1.0/w) * crit(pred, t)
	return loss