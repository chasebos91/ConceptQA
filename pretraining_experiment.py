from soft_CE_loss import *
from Concept_Module import *
from Noam import NoamOptim
from batching import *
torch.manual_seed(0)

cat_data = pickle.load(open("cat_data", "rb"))
cat_ans = pickle.load(open("cat_ans_key", "rb"))

vdata = pickle.load(open("vanilla_data", "rb"))
vans = pickle.load(open("vanilla_ans", "rb"))

sdata = pickle.load(open("soft_data", "rb"))
sans = pickle.load(open("soft_ans", "rb"))

num_points = len(cat_data)
cat_data = sort_data(cat_data)
batches = batch(cat_data, 200, num_points)

modality_dims = {"audio":[36, 20, 10], "flow":[8,6, 4], "haptics" :[13,10, 5],"vibro":[70,50, 30],
		  "surf":[90,60, 40], "fingers":[6,4, 3], "color":[134,100, 50], "weight": [1, 1, 1], "height": [1,1,1]}

model_file = "conceptqa_pretrained"
epochs = 1
pt_model = ConceptQA(modality_dims, 768, 200, 100, pretraining=True, num_cats=len(cat_ans))

crit = nn.CrossEntropyLoss()
lr = 5.0

#optim = torch.optim.Adam(conceptqa.parameters(), 1e-3)
optim = NoamOptim(pt_model.parameters(), 100)
#scheduler = torch.optim.lr_scheduler.StepLR(optim, 1, gamma=0.95)

prev = np.float("inf")
#remove
batches = [[batches[0][0]]]
cat_ll = []
for e in range(epochs):
	start = time.time()
	for b in batches:
		cat_ll.append(pretrain(pt_model, b, optim, crit))
	#TODO:make train and val REMEMBER to remove the index on batches
	val_loss, preds = evaluate(pt_model, batches[0], crit, pt=True)
	if val_loss < prev:
		prev = val_loss
		best_model = pt_model
	#scheduler.step()
	elapsed = time.time() - start
	print("Epoch:", e, val_loss)
	start = time.time()


torch.save(best_model.state_dict(), model_file)

vanilla_model = copy.deepcopy(best_model)
soft_model = copy.deepcopy(best_model)
vanilla_model.fine_tune()
soft_model.fine_tune()

v = sort_data(vdata)
vanilla_batches = batch(v, 200, len(vdata))
s = sort_data(sdata)
soft_batches = batch(s, 200, len(sdata))
s_optim = NoamOptim(soft_model.parameters(), 100)
v_optim = NoamOptim(vanilla_model.parameters(), 100)
soft_crit = soft_cross_entropy
s_prev = np.float("inf")
v_prev = np.float("inf")

##remove
vanilla_batches = [[vanilla_batches[0][0]]]
soft_batches = [[soft_batches[0][0]]]
s_ll = []
v_ll = []
for e in range(epochs):
	start = time.time()
	for s, v in zip(soft_batches, vanilla_batches):

		v_ll.append(train(vanilla_model, v, v_optim, crit))
		s_ll.append(train(soft_model, s, s_optim, soft_crit, soft=True))
	#T
	v_val_loss, v_preds = evaluate(vanilla_model, v, crit)
	s_val_loss, s_preds = evaluate(soft_model, s, soft_crit, soft=True)
	if v_val_loss < v_prev:
		prev = v_val_loss
		v_best_model = vanilla_model
	if s_val_loss < s_prev:
		prev = s_val_loss
		s_best_model = soft_model
	# scheduler.step()
	elapsed = time.time() - start
	print("Epoch:", start, v_val_loss)
	print("Epoch:", start, s_val_loss)
	start = time.time()


s_model_file = open("finetuned_s_model", "wb")
v_model_file = open("finetuned_v_model", "wb")

torch.save(s_best_model.state_dict(), s_model_file)
torch.save(v_best_model.state_dict(), v_model_file)
#save loss curves
pickle.dump(cat_ll, open("pt_exp_cat_loss", "wb"))
pickle.dump(s_ll, open("pt_exp_s_loss", "wb"))
pickle.dump(v_ll, open("pt_exp_v_loss", "wb"))
