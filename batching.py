import random

def batch(data, batch_size, data_len):
	num_batches = int(data_len/batch_size)
	sub_batch_size = int(batch_size/3)
	yes, no, descriptive = data[0], data[1], data[2]
	batches = []
	for i in range(num_batches):

		temp = random.sample(yes, int(sub_batch_size/2)) + random.sample(no, int(sub_batch_size/2)) + random.sample(descriptive, int(sub_batch_size*2))
		random.shuffle(temp)
		batches.append(temp)


	return batches


def sort_data(data):
	yes, no, descriptive = [], [], []
	for d in data:
		if d[3] == "yes":
			yes.append(d)
		elif d[3] == "no":
			no.append(d)
		else: descriptive.append(d)

	return [yes, no, descriptive]