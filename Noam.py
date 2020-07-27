from torch.optim import Adam
#https://github.com/dreamgonfly/Transformer-pytorch/blob/master/optimizers.py

class NoamOptim(Adam):

	def __init__(self, params, dim, factor=2, wu_steps = 1000, betas=(.9, .98), eps=1e-9):
		self.dim = dim
		self.wu_steps = wu_steps
		self.lr = 0
		self.steps = 0
		self.factor = factor

		super(NoamOptim, self).__init__(params, betas=betas, eps=eps)

	def step(self, closure=None):
		self.steps +=1
		self.lr = self.lrate()
		for g in self.param_groups:
			g['lr'] = self.lr
		super(NoamOptim, self).step()

	def lrate(self):
		return self.factor * self.dim ** (-.5) * min(self.steps ** -.5, self.steps * self.wu_steps ** -1.5)