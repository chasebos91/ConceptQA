import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.modules.transformer as t
from torch.nn.utils.weight_norm import weight_norm
from torch.optim import Adam
from Transformer import *
from Question_Module import QuestionEmbed
import pickle
import spacy
#Query aware attention encoder?
from transformers import (OpenAIGPTDoubleHeadsModel, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                                  GPT2DoubleHeadsModel, GPT2LMHeadModel, GPT2Tokenizer)

class ConceptQA(nn.Module):
	def __init__(self, tuples, q_dim, hid_dim, emb_dim, num_cats=None, pretraining=False, coattn=True):
		super().__init__()
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		#print('Using device:', self.device)
		self.vocab = pickle.load(open("vocab", "rb"))
		#self.vocab = process_answer_corpus()
		self.idx_to_w = dict((y,x) for x,y in self.vocab.items())
		self.init_network(tuples)
		self.hid_dim = hid_dim
		self.Q_embed = QuestionEmbed()
		self.q_proj = nn.Linear(q_dim, emb_dim)
		self.MM_embed = nn.Linear(432, emb_dim)
		self.max_len = 11
		self.decoder = TransformerModel(len(self.vocab), emb_dim, pretraining)
		self.coattn = coattn
		if not coattn:
			#double check concat mm dim
			self.MM_embed = nn.Linear(144, emb_dim)
		if pretraining:
			self.pred_layers = nn.Sequential(nn.Linear(1100, 550),
			                                 nn.LeakyReLU(), nn.Linear(550, num_cats) ).to(self.device)
			self.pretrain = pretraining


	def init_network(self, tuples):
		temp = {}

		for k in tuples.keys():
			temp[k] = Encoder(tuples[k][0], tuples[k][1], tuples[k][2], self.device)
			#nn.init.uniform_(temp[k].weight, -initrange, initrange)
		self.encoders = nn.ModuleDict(temp)

	def init_weights(self):
		initrange = 0.1
		nn.init.uniform_(self.encoders.weight, -initrange, initrange)
		nn.init.zeros_(self.decoder.weight)
		nn.init.uniform_(self.decoder.weight, -initrange, initrange)
		"""
		self.decoder = OpenAIGPTDoubleHeadsModel.from_pretrained('openai-gpt')
		self.tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
		self.tokenizer.set_special_tokens(self.special_tokens)
		self.decoder.set_num_special_tokens(len(self.special_tokens))
		"""
	def fine_tune(self):
		self.pred_layers.requires_grad = False
		self.pretrain = False


	def parallel_co_attention(self, Q, E):
		# https://github.com/atulkum/co-attention/blob/master/code/model.py
		Q = F.tanh(self.q_proj(Q))
		#.view(-1, self.hid_dim))).view(Q.size())
		#E_t = torch.transpose(E, 1, 2)
		if len(Q.shape) == 2:
			L = torch.mul(E, Q)
			A_Q = F.softmax(L, dim=0)
			E = E.T
			C_Q = torch.mm(E, A_Q)
			A_E = F.softmax(L.T, dim=0)
			C_E = torch.mm(torch.cat((Q, C_Q)), A_E)
			summary = torch.cat((C_E, E)).reshape((-1, 432))
			return self.MM_embed(summary)
		else: L = torch.bmm(Q, E)
		A_Q = F.softmax(L, dim=1)
		#A_Q = torch.transpose(A_Q, 1, 2)
		C_Q = torch.bmm(E, A_Q)
		Q_t = torch.transpose(Q, 1, 2)
		A_E = F.softmax(L, dim=2)
		C_E = torch.bmm(torch.cat((Q_t, C_Q), 1 ), A_E)
		C_E_t = torch.transpose(C_E, 1, 2)
		return self.MM_embed(torch.cat((C_E_t, E), 2))

	def build_input(self, q_emb, act_embs, answer=None):
		input = [self.special_tokens[0], q_emb]
		for i in range(act_embs):
			input += [self.special_tokens[i + 1], act_embs[i]]
		input += [self.special_tokens[len(self.special_tokens)], answer]


	def forward(self, question, object):

		codes = []

		q, attn_mask = self.Q_embed.process(question)
		Q = self.Q_embed(q, attn_mask)
		i = 0
		for action in object.values():
			modalities = []
			#
			#if i != 10:
			for modality in action:
				feat = torch.tensor(action[modality], dtype=torch.float).to(self.device)
				print(feat.cuda())

				x = self.encoders[modality](feat)
				if len(x.shape) == 1:
					x = x.unsqueeze(dim=0)
				elif len(x.shape) == 3:
					x = x.reshape(1, 1)
				x = x.permute(1, 0)

				modalities.append(x)
			i += 1

			code = torch.cat(modalities)
			if self.coattn:
				code = self.parallel_co_attention(Q, code)
			else: code = self.MM_embed(code)
			codes.append(code)
		Q = F.tanh(self.q_proj(Q))
		codes = torch.stack([Q] + codes)
		codes = codes.permute(1, 0, 2)


		if self.pretrain:
			src = self.decoder.pos_encoder(codes)
			out = self.decoder.transformer_encoder(src, None)
			out = out.flatten(0)
			out = self.pred_layers(out)
			return out
			#return F.log_softmax(out, dim=-1)
		out = self.decoder(codes)
		return out






class Encoder(nn.Module):
	def __init__(self, in_dim, hid_dim, out_dim, device):
		super().__init__()
		self.net = nn.Sequential(weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
		                          nn.ReLU(),
		                          weight_norm(nn.Linear(hid_dim, out_dim), dim=None), nn.ReLU())
		self.device = device

		self.net.apply(self.init_weights)

	def forward(self, x):
		return self.net(x.to(self.device))

	def init_weights(self, m):
		initrange = 0.1
		if type(m) == nn.Linear:
			#m.weight.data.fill_(1.0)
			nn.init.uniform_(m.weight, -initrange, initrange)



def build_idx_to_word(corpus):
	idx_to_word = {}
	word_to_idx = {}
	for w, i in enumerate(corpus):
		if w not in word_to_idx:
			idx_to_word[i] = w
			word_to_idx[w] = i
	return idx_to_word, word_to_idx

def build_targets(answer, vocab, max_len):

	ans_vector = np.zeros(max_len)
	for i, w in enumerate(answer):
		ans_vector[i] = vocab[w]
	return np.array(ans_vector, dtype=int)


def idx_to_word(preds, idx_2_w):
	words = []
	if torch.is_tensor(preds):
		preds = preds.numpy()[0]
	for p in preds:
		words.append(idx_2_w[int(p)])
	return words
	#for w in row, argmax the idx and pass idx to idx_to_word

def process_answer_corpus():
	vocab = {"pad": 0}
	answers = pickle.load(open("vanilla_ans", "rb"))
	tokenizer = spacy.load("en_core_web_sm")
	i = 1
	for ans in answers:
		ans = tokenizer(ans)
		ans = [token.text for token in ans]
		for w in ans:
			if w not in vocab:
				vocab[w] = i
				i += 1
	pickle.dump(vocab, open("vocab", "wb"))
	return vocab

def pretrain(net, data, optim, crit):
	net.train()
	total_loss = 0
	start = time.time()

	i = 1
	for d in data:
		q, feats, targets = d[0], d[1][0][1], d[2]
		targets = torch.tensor(targets).unsqueeze(0).to(net.device)
		if "labels" in feats.keys():
			feats.pop("labels")
		net.zero_grad()
		preds = net(q, feats)

		preds = preds.unsqueeze(0)
		loss = crit(preds, targets)
		loss.backward()
		torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
		optim.step()
		# think about clipping gradients
		total_loss += loss.item()
		i += 1
		"""
		if i % 200 == 0 and i > 0:
			cur_loss = total_loss / 200
			elapsed = time.time() - start
			temp = F.softmax(preds, dim=1)
			_, idxs = temp.max(dim=1)

			print("Training\n", "time: ", elapsed, "s\n", "running loss: ", cur_loss)
			print(loss.item())
			total_loss = 0
			start = time.time()"""
	cur_loss = total_loss / 200
	elapsed = time.time() - start
	temp = F.softmax(preds, dim=1)
	_, idxs = temp.max(dim=1)

	print("Training\n", "time: ", elapsed, "s\n", "running loss: ", cur_loss)
	print(loss.item())



	return net, cur_loss

def train(net, data, optim, crit, soft=False):

	#optim = Adam(net.parameters(), 1e-3)
	net.train()
	total_loss = 0
	start = time.time()
	toks = len(net.vocab)
	loss_list = []


	i = 1
	for d in data:
		q, feats, targets = d[0], d[1][0][1], d[2]
		if "labels" in feats.keys():
			feats.pop("labels")
		if soft:
			target_list = []
			for t in targets:
				target_list.append(build_targets(t, net.vocab, net.max_len))
			targets = target_list
		else: targets = build_targets(targets, net.vocab, net.max_len)
		net.zero_grad()
		preds_ = net(q, feats)

		preds = preds_.view(-1, toks)
		loss = crit(preds, torch.tensor(targets))
		loss.backward()
		torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
		optim.step()
		#think about clipping gradients
		total_loss += loss.item()
		i += 1
		"""
		if i % 200 == 0 and i > 0:
			cur_loss = total_loss / 200
			elapsed = time.time() - start
			temp = F.softmax(preds_, dim=2)
			_, idxs = temp.max(dim=2)
			words = idx_to_word(idxs, net.idx_to_w)
			answer = idx_to_word(targets, net.idx_to_w)
			print("Training\n", "time: ",elapsed, "s\n", "running loss: ", cur_loss)
			print(loss.item())
			print("Q:", q)
			print("A:", answer)
			print("Pred:", words)
			total_loss = 0
			start = time.time()"""
	cur_loss = total_loss / 200
	elapsed = time.time() - start
	temp = F.softmax(preds_, dim=2)
	_, idxs = temp.max(dim=2)
	words = idx_to_word(idxs, net.idx_to_w)
	# TODO: copy this over to evaluate (or just replace it)
	if soft:
		answer = []
		for t in targets:
			answer.append(idx_to_word(t, net.idx_to_w))

	else: answer = idx_to_word(targets, net.idx_to_w)
	print("Training\n", "time: ", elapsed, "s\n", "running loss: ", cur_loss)
	print(loss.item())
	print("Q:", q)
	print("A:", answer)
	print("Pred:", words)
	return cur_loss

def evaluate(net, data, crit, pt=False, soft=False):
	net.eval()
	total_loss = 0
	start = time.time()
	toks = len(net.vocab)
	#data = data[:int(len(data)/5)]

	i = 0
	preds_list = []
	with torch.no_grad():
		for d in data:
			q, feats, targets = d[0], d[1][0][1], d[2]
			if "labels" in feats.keys(): feats.pop("labels")

			net.zero_grad()
			preds = net(q, feats)
			if not pt:
				if soft:
					target_list = []
					for t in targets:
						target_list.append(build_targets(t, net.vocab, net.max_len))
					targets = target_list
				else: targets = build_targets(targets, net.vocab, net.max_len)
				preds = preds.view(-1, toks)

			preds_list.append(preds)
			preds = preds.unsqueeze(0)

			if pt: loss = crit(preds, torch.tensor(targets).unsqueeze(0))
			else: loss = crit(preds.squeeze(0), torch.tensor(targets))
			# think about clipping gradients
			total_loss += loss.item()
			i += 1


	cur_loss = total_loss / 100
	elapsed = time.time() - start
	print("Validation\n", elapsed, "s\n", "running loss: ", cur_loss)
	print(loss.item())

	return cur_loss, preds_list


