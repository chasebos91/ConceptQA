import numpy as np
import random
from collections import defaultdict
import os
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import spacy



def build_feature_set():
	sensorimotor = ["crush_audio.txt", "crush_flow.txt", "crush_haptics.txt", "crush_surf.txt", "crush_vibro.txt",
	                "grasp_audio.txt",
	                "grasp_fingers.txt", "grasp_flow.txt", "grasp_haptics.txt", "grasp_surf.txt", "grasp_vibro.txt",
	                "hold_audio.txt",
	                "hold_flow.txt", "hold_surf.txt", "hold_vibro.txt", "lift_slow_audio.txt", "lift_slow_flow.txt",
	                "lift_slow_surf.txt",
	                "lift_slow_vibro.txt", "look_color.txt", "look_surf.txt", "low_drop_audio.txt", "low_drop_flow.txt",
	                "low_drop_haptics.txt", "low_drop_surf.txt", "low_drop_vibro.txt", "poke_audio.txt",
	                "poke_haptics.txt", "poke_surf.txt", "poke_vibro.txt", "push_audio.txt", "push_flow.txt",
	                "push_haptics.txt", "push_surf.txt",
	                "push_vibro.txt", "shake_audio.txt", "shake_flow.txt", "shake_haptics.txt", "shake_surf.txt",
	                "shake_vibro.txt", "tap_audio.txt",
	                "tap_flow.txt", "tap_haptics.txt", "tap_surf.txt", "tap_vibro.txt"]

	path = "Data/dataset/cy101/"
	rel_path = "Data/dataset/cy101/objects.txt"
	smp = os.path.dirname(os.path.dirname(__file__)) + "/Thesis/Data/dataset/cy101/sensorimotor_features/"

	# generate 5 different pca
	sensor_dict = {"audio": [], "flow": [], "haptics": [], "surf": [], "vibro": [], "color": []}
	actions = ["crush" ,"look", "grasp", "hold", "lift", "drop", "poke", "push", "shake", "tap"]
	feature_types = ["audio", "flow", "haptics" ,"vibro", "surf", "fingers", "color", "weight", "height"]
	pca_dict = {"audio": PCA(n_components = 0.99), "flow": PCA(n_components = 0.99), "haptics": PCA(n_components = 0.99),
	            "surf": PCA(n_components = 0.99), "vibro": PCA(n_components = 0.99), "color": PCA(n_components = 0.99)}


	for file in sensorimotor:
		rel_path = smp + file
		fp = os.path.join(smp, rel_path)
		with open(fp, "r") as f:
			for k in pca_dict.keys():
				if k in file:
					for l in f:
						feats = l.split(",")[1:]
						feats = list(map(float, feats))
						if not np.isnan(feats).any():
							sensor_dict[k].append(feats)

	scalers = {}

	for k in pca_dict.keys():

		scaler = MinMaxScaler()

		scaled_data = scaler.fit_transform(np.asarray(sensor_dict[k]))

		scalers[k] = scaler
		pca_dict[k].fit(np.asarray(scaled_data))

	csv = "cy101_labels.csv"

	# get objects
	"""
	objects = defaultdict()
	with open(rel_path) as obj:
		lines = obj.readlines()
		for l in lines:
			#temp = l.split(",")[0]
			obj = l.strip()

			objects[obj] = defaultdict()"""

	objectnames = []
	with open(rel_path) as obj:
		lines = obj.readlines()
		for l in lines:
			temp = l.split(",")[0]
			obj = l.strip()

			objectnames.append(obj)

	objects = defaultdict()
	feat_names = []
	for f in sensorimotor:
		rel_path = smp + f
		fp = os.path.join(smp, rel_path)
		with open(fp) as feat:
			for l in feat:
				obj = l.split(",")[0]
				feats = l.split(",")[1:]

				if obj not in objects:
					objects[obj] = defaultdict(defaultdict)

				for a in actions:
					if a in f:
						action = a
				for s in feature_types:
					if s in f:
						sensor = s
				#temp = f.replace(".txt", "")
				if action not in feat_names: feat_names.append(action)
				feats = list(map(float, feats))
				if not np.isnan(feats).any():
					for n in scalers.keys():
						if n in f:
							temp = scalers[n].transform(np.asarray(feats).reshape(1,-1))
							feats = pca_dict[n].transform(temp)
					objects[obj][action][sensor] = feats


	rem = []

	for k, v in objects.items():
		if len(v) != len(feat_names):
			rem.append(k)

	wh_list = []
	weight = []
	height = []
	with open(path + "objects_weight_height.txt") as f:
		lines = f.read().splitlines()
		lines = lines[1:][:]
		for l in lines:
			t = l.split("\t")
			w = t[1]
			h = t[2]
			for o in objects.keys():
				if t[0] in o:
					weight.append(float(w))
					height.append(float(h))

	weight_scale, height_scale = MinMaxScaler(), MinMaxScaler()
	weight_scale.fit(np.asarray(weight).reshape(-1, 1))
	height_scale.fit(np.asarray(height).reshape(-1, 1))



	with open(path + "objects_weight_height.txt") as f:
		lines = f.read().splitlines()
		lines = lines[1:][:]
		for l in lines:
			t = l.split("\t")
			w = t[1]
			h = t[2]
			for o in objects.keys():
				if t[0] in o:
					objects[o]["lift"]["weight"] = [weight_scale.transform(np.asarray(float(w)).reshape(-1, 1))]
					objects[o]["look"]["height"] = [height_scale.transform(np.asarray(float(h)).reshape(-1, 1))]

	with open(path + csv) as f:
		lines = f.read().splitlines()
		lines = lines[1:][:]

		for l in lines:
			t = l.split(",")
			name = t[0].replace("_", " ")
			for o in objects.keys():
				if t[0] in o:
					objects[o]["labels"] = []

					for w in t:
						w = w.replace('\"', "")
						w = w.replace(' ', "")
						objects[o]["labels"].append(w)
				#objects[o]["labels"].append(name)

	# remove no_object
	for r in rem:
		objects.pop(r)

	feat_sizes = {"audio":36, "flow":8, "haptics" :13,"vibro":70,
          "surf":90, "fingers":6, "color":134, "weight":1,
          "height":1}

	for obj in objects.values():
		for act in obj.keys():
			features = obj[act]
			for feat in feature_types:
				if feat not in features and act != 'labels':
					size = feat_sizes[feat]
					features[feat] = np.zeros(size)

	return objects
	# get object names

"""

	obj_names = {}
	for k in objects.keys():
		# h, s, t = k.partition("_")
		h = k.replace("_", " ")
		if h != "no object":
			if h not in obj_names:
				obj_names[h] = []
			obj_names[h] = objects[k]
	return obj_names

	obj_list = []
	for k in objects.keys():
		if temp[-1].isdigit():
			if temp[-2] == "t":
				num = "_t" + temp[-1]
				obj = temp.replace(num, "")
			else:
				num = "_" + temp[-1]
				obj = temp.replace(num, "")
		obj_list.append([obj, objects[obj]])
"""
# if split word part two is not number, concatenate
# make subsection of objects for task (cup, ball, bottle, cone, weight, tupperware)
# before make tuple and sentence at same time

def get_color_label(obj):
	colors = ["red", "yellow", "purple", "pink", "brown", "tan", "green", "multicolored", "orange"]
	labels = obj["labels"]
	for l in labels:
		if l in colors:
			return l

def is_name(a, obj_names):
	names = list(obj_names.keys())
	if a in names: return True
	return False


def same_type(x, y):
	tempx = x.partition(" ")
	tempy = y.partition(" ")
	if tempx == tempy: return True
	return False


def make_names_normal(name):
	names = {"bigstuffedanimal": "big stuffed animal", "weight": "weight", "pvc": "pvc",
	         "smallstuffedanimal": "small stuffed animal", "noodle": "noodle", "eggcoloringcup": "egg coloring cup",
	         "cone": "cone", "cannedfood": "canned food"}
	n, _, _ = name.partition(" ")
	if n in names.keys():
		return names[n]
	# split the names by space

	if n == "metal":
		return name.strip(n)
	return n


def get_answer(obj, adj):
	negative = ["It is not.", "It isn't.", "Nope.", "No."]
	positive = ["Yes it is.", "It is.", "Yep.", "Yes."]

	labels = obj["labels"]
	if adj in labels:
		return random.choice(positive)
	else:
		return random.choice(negative)


def get_adj():
	adjectives = []
	with open("Data/dataset/cy101/cy101_labels.csv") as f:
		lines = f.read().splitlines()
		lines = lines[1:][:]
		for l in lines:
			t = l.split(",")

			for w in t:
				if len(w) > 1:
					w = w.replace('\"', "")
					w = w.replace("_", " ")
					if w[0] == " ": w = w[1:]
					if w not in adjectives: adjectives.append(w)

	return adjectives



def single_gen(obj_names):
	# either q type 1 or 2

	adj = get_adj()
	tokenizer = spacy.load("en_core_web_sm")

	tuples = []
	answers = ["yes", "no"]
	#TODO: include contents!!

	categories = {"material": ["plastic", "rubber", "metal", "wicker", "foam"],
	              "texture": ["hard", "soft", "squishy"], "shape": ["cylindrical", "round", "cone", "rectangular"],
	              "size": ["small", "large", "wide"], "weight": ["heavy", "light"]}
	adj_list = ["plastic", "rubber", "metal", "wicker", "foam", "hard", "soft", "squishy", "cylindrical", "round", "cone", "rectangular",
	            "small", "large", "wide", "heavy", "light"]

	ans_formats = ["The object is ", "It's "]
	category_formats = [["The ", " of the object is "], ["The object ", " is "], ["The ", " is "]]

	# TODO make sure that the object has its name (k, v)

	# one type
	for o in obj_names.keys():
		# need to change object names that are weird to natural text
		name = make_names_normal(o)
		q_format = ["What color is the object?", "What's the object's color?", "Describe the object color.", "Describe the color of the object.", "What's the color?"]
		color = get_color_label(obj_names[o])
		color_ans = ans_formats + ["The color of the object is ", "The object's color is ", "The color is "]

		if color != None:
			for q in q_format:
				for a in color_ans:
					ans = a + color + "."
					ans1 = tokenizer(ans)
					ans1 = [token.text for token in ans1]
					tuples.append([q, [(o, obj_names[o])], ans1])
					if ans not in answers:
						answers.append(ans)


		q_format = [["What is the ", " of the object?"], ["What's the object's ", "?"], ["What's the ", " of the object?"], ["Describe the ", " of the object."], ["Describe the ", "."], ["What's the ", "?"]]
		for c in categories.keys():
			adjs = categories[c]

			for a in adjs:
				if a in obj_names[o]["labels"]:
					for question in q_format:

						for cq in category_formats:
							ans = cq[0] + c + cq[1] + a + "."
							ans1 = tokenizer(ans)
							ans1 = [token.text for token in ans1]
							q = question[0] + c + question[1]
							tuples.append([q, [(o, obj_names[o])], ans1])
							if ans not in answers: answers.append(ans)

						for af in ans_formats:
							ans = af + a + "."
							ans1 = tokenizer(ans)
							ans1 = [token.text for token in ans1]
							q = question[0] + c + question[1]
							tuples.append([q, [(o, obj_names[o])], ans1])
							if ans not in answers: answers.append(ans)


		q_format = ["Is the object heavy or light?", "What is the weight of the object?", "What's the object's weight?", "How heavy is the object?", "Describe the object's weight.", "Describe the object's heaviness.", "What's the weight of the object?"]
		if "heavy" in obj_names[o]["labels"] or "light" in obj_names[o]["labels"]:
			for q in q_format:
				if "heavy" in obj_names[o]["labels"]: a = "heavy"
				else: a = "light"

				for cq in category_formats:
					ans = cq[0] + "weight" + cq[1] + a + "."
					ans1 = tokenizer(ans)
					ans1 = [token.text for token in ans1]
					tuples.append([q, [(o, obj_names[o])], ans1])
					if ans not in answers: answers.append(ans)

				for af in ans_formats:
					ans = af + a + "."
					ans1 = tokenizer(ans)
					ans1 = [token.text for token in ans1]
					tuples.append([q, [(o, obj_names[o])], ans1])
					if ans not in answers: answers.append(ans)


		# another type
		negative = ["It is not.", "It isn't.", "Nope.", "No."]
		positive = ["Yes it is.", "It is.", "Yep.", "Yes."]



		#Get rid of all adjectives which are actually names
		q_format = [["Is the object ","?"], ["Would you describe the object as ", "?"]]
		i = 0
		for question in q_format:
			for a in adj_list:
				q = question[0] + a + question[1]
				if a in adj_list:
					obj, _, _ = o.partition(" ")
					labels = obj_names[o]["labels"]
					if adj in labels:

						for ans in positive:
							ans1 = tokenizer(ans)
							ans1 = [token.text for token in ans1]
							i += 1
							tuples.append([q, [(o, obj_names[o])], ans1])
							if ans not in answers: answers.append(ans)

					else:
						for ans in negative:
							ans1 = tokenizer(ans)
							i += 1
							ans1 = [token.text for token in ans1]
							tuples.append([q, [(o, obj_names[o])], ans1])
							if ans not in answers: answers.append(ans)

	print(i)
	print(len(tuples))
	return tuples, answers


def build_dataset():
	obj_names = build_feature_set()
	#tp, ap = pair_gen(obj_names)
	ts, asingle = single_gen(obj_names)
	print("single tupes: ", len(ts))
	#print("pair tupes: ", len(tp))
	#tuples = tp + ts
	#answers = asingle + ap
	sentences = []
	for s in ts:
		if s[0] not in sentences:
			sentences.append(s[0])
	return ts, asingle,  sentences

dataset, answers, corpus = build_dataset()
random.shuffle(dataset)

dfile = "datafile"
ans = "ansfile"
c = "corpus"
ofd = open(dfile, "wb")
ofa = open(ans, "wb")
ofc = open(c, "wb")

pickle.dump(dataset, ofd)
pickle.dump(answers, ofa)
pickle.dump(corpus, ofc)

ofd.close()
ofa.close()
ofc.close()

data = pickle.load(open("datafile", "rb"))
ans = pickle.load(open("ansfile", "rb"))
a = data[0]


