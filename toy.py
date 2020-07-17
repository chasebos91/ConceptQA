from transformers import BertModel, BertTokenizer
import warnings
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# run block of code and catch warnings
with warnings.catch_warnings():
	# ignore all caught warnings
	warnings.filterwarnings("ignore")

print(torch.__version__)


class SSTDataset(Dataset):
	def __init__(self, filename, maxlen):
		self.df = pd.read_csv(filename, delimiter="\t")
		self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
		self.maxlen = maxlen

	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		sentence = self.df.loc[idx, 'sentence']
		label = self.df.loc[idx, 'label']
		toks = self.tokenizer(sentence)
		toks = ['[CLS]'] + toks + ['[SEP]']

		if len(toks) < self.maxlen:
			toks = toks + ['[PAD]' for _ in range(self.maxlen - len(toks))]
		else:
			toks = toks[:self.maxlen-1]

		tok_ids = self.tokenizer.convert_tokens_to_ids(toks)
		tok_ids_tensor = torch.tensor(tok_ids)
		attn_mask = (tok_ids_tensor != 0).long()

		return tok_ids_tensor, attn_mask, label

def toy_ex():

	bert_m = BertModel.from_pretrained('bert-base-uncased')
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

	#TOKENIZE
	ex = "I really enjoyed this movie a lot."
	toks = tokenizer.tokenize(ex)
	print(toks)

	#ADD SPECIAL TOKENS
	toks = ['[CLS]'] + toks + ['[SEP]']

	#PADD INPUT
	T = 12
	padded = toks + ['[PAD]' for _ in range(T-len(toks))]
	print(padded)


	#makes bert only focus on words and not padding tokens
	attn_mask = [1 if tok != '[PAD]' else 0 for tok in padded]
	print(attn_mask)

	#when we have multiple inputs, we maintain a list of segment tokens
	#such that each part of the input corresponds to a value (0, 1, 2 etc)
	seg_ids = [0 for _ in range(len(padded))]
	# this line above would be different if there were more than one input to bert
	# this example has only one

	#obtain the id's corresponding to the tokens
	sent_ids = tokenizer.convert_tokens_to_ids(padded)
	print(sent_ids)

	#put everything to torch tensor

	tok_ids = torch.tensor(sent_ids).unsqueeze(0)
	attn_mask = torch.tensor(attn_mask).unsqueeze(0)
	seg_ids = torch.tensor(seg_ids).unsqueeze(0)

	hids, cls_head = bert_m(tok_ids, attention_mask = attn_mask, token_type_ids = seg_ids)
	a = 0

class SentimentClassifier(nn.Module):
	def __init__(self, freeze=True):
		super(SentimentClassifier, self).__init__()
		self.bert_lay = BertModel.from_pretrained('bert-base-uncased')
		if freeze:
			for p in self.bert_lay.named_parameters():
				p.requires_grad = False

		self.cls_layer = nn.Linear(768, 1)

	def forward(self, seq, attn_masks):
		cont_reps, _ = self.bert_lay(seq, attn_masks)
		cls_rep = cont_reps[:, 0]
		return self.cls_layer(cls_rep)

def experiment():
	#using the dataloader for our wrapped dataset class
	train_set = SSTDataset(filename = 'data/SST-2/train.tsv', maxlen = 30)
	val_set = SSTDataset(filename = 'data/SST-2/dev.tsv', maxlen = 30)

	#Creating intsances of training and validation dataloaders
	train_loader = DataLoader(train_set, batch_size = 64, num_workers = 5)
	val_loader = DataLoader(val_set, batch_size = 64, num_workers = 5)

	net = SentimentClassifier()

	import torch.optim as optim

	crit = nn.BCEWithLogitsLoss()
	opt = optim.Adam(net.parameters(), lr=2e-5)

def train(net, crit, opt, train_loader, val_loader, args):

	pass

toy_ex()