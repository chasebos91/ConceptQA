from transformers import DistilBertTokenizer, DistilBertModel
import torch
import numpy as np
import pickle
import warnings
import torch.nn as nn
with warnings.catch_warnings():
	# ignore all caught warnings
	warnings.filterwarnings("ignore")
"Reference: http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/"

class QuestionEmbed(nn.Module):
	def __init__(self):
		super(QuestionEmbed, self).__init__()
		self.model = DistilBertModel.from_pretrained('distilbert-base-uncased-distilled-squad')
		self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-distilled-squad')
		self.max_len = 17

	def process(self, sentences):
		#tokenize and pad text
		tok_text = self.tokenizer.encode(sentences, add_special_tokens=True)
		pad = [0] * (self.max_len - len(tok_text))
		pad = tok_text + pad
		#pad = np.array([i + [0] * (self.max_len - i) for i in tok_text])
		attn_mask = []
		for tok in pad:
			if tok != 0: attn_mask += [1]
			else: attn_mask += [0]

		attn_mask = torch.tensor(attn_mask)
		x = torch.tensor(pad)

		return x, attn_mask


	def forward(self, x, attn_mask):
		x = x.unsqueeze(0)
		out = self.model(x, attention_mask=attn_mask)
		embeddings = out[0][:, 0, :]
		return embeddings


def unit_test():
	data = pickle.load(open("datafile", "rb"))

	d = data[0][0]
	clf = QuestionEmbed()
	x, attn = clf.process(d)
	emb = clf.embed(x, attn)
	a = 0

