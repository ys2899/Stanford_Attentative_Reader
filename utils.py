import pdb
from collections import Counter
import logging
import lasagne
import numpy as np


def load_data(in_file, max_example=None, relabeling=True):
	documents = []
	questions = []
	answers = []
	num_examples = 0
	f = open(in_file, 'r')

	while True:
		line = f.readline()
		if not line:
			break

		question = line.strip().lower()
		answer = f.readline().strip()
		document = f.readline().strip().lower()

		if relabeling:
			q_words = question.split(' ')
			d_words = document.split(' ')

			assert answer in d_words

			entity_dict = {}
			entity_id = 0

			for word in d_words + q_words:
				if word.startswith('@entity') and (word not in entity_dict):
					entity_dict[word] = '@entity' + str(entity_id)
					entity_id += 1

			q_words = [entity_dict[w] if w in entity_dict else w for w in q_words]
			d_words = [entity_dict[w] if w in entity_dict else w for w in d_words]
			answer = entity_dict[answer]

			question = ' '.join(q_words)
			document = ' '.join(d_words)
		questions.append(question)
		answers.append(answer)
		documents.append(document)
		num_examples += 1
		f.readline()
		if max_example is not None and num_examples >= max_example:
			break
	f.close()

	return (documents, questions, answers)

def build_dict(sentences, max_words=50000):
	wc = Counter()
	for s in sentences:
		for w in s.split(' '):
			wc[w] += 1

	ls = wc.most_common(max_words)
	logging.info('# of Words: %d -> %d' % (len(wc), len(ls)))
	for k in ls[:5]:
	    logging.info(k)
	logging.info('...')
	for k in ls[-5:]:
	    logging.info(k)
	# Build a dictionary
	return {w[0]: i+2 for (i,w) in enumerate(ls)}



def prepare_data(seqs):

	lengths = [len(seq) for seq in seqs]
	n_samples = len(seqs)
	max_len = np.max(lengths)
	x = np.zeros((n_samples, max_len)).astype('int32')
	x_mask = np.zeros((n_samples, max_len)).astype('float32')

	for idx, seq in enumerate(seqs):
		x[idx, :lengths[idx]] = seq
		x_mask[idx, :lengths[idx]] = 1.0

	return x, x_mask


def get_minibatches(n, minibatch_size, shuffle=False):
	idx_list = np.arange(0, n, minibatch_size)
	if shuffle:
		np.random.shuffle(idx_list)
	minibatches = []
	for idx in idx_list:
		minibatches.append(np.arange(idx, min(idx + minibatch_size, n)))
	return minibatches


def gen_embeddings(word_dict, dim, in_file=None,
				   init=lasagne.init.Uniform()):

	num_words = max(word_dict.values()) + 1
	embeddings = init((num_words, dim))

	if in_file is not None:
		pre_trained = 0
		for line in open(in_file).readlines():
			sp = line.split()
			if sp[0] in word_dict:
				pre_trained += 1
				embeddings[word_dict[sp[0]]] = [float(x) for x in sp[1:]]

	return embeddings



def vectorize(examples, word_dict, entity_dict,
			  sort_by_len=True, verbose=True):
	
	"""
		vectorize 'examples'.
		in_x1, in_x2: sequences for document and question respectively.
		in_y: label
		in_l: whether the entity label occurs in the document.
	"""

	in_x1 = []
	in_x2 = []
	in_l = np.zeros((len(examples[0]), len(entity_dict))).astype(float)
	in_y = []

	for idx, (d, q, a) in enumerate(zip(examples[0], examples[1], examples[2])):
		d_words = d.split(' ')
		q_words = q.split(' ')


		assert(a in d_words)

		seq1 = [word_dict[w] if w in word_dict else 0 for w in d_words]
		seq2 = [word_dict[w] if w in word_dict else 0 for w in q_words]

		# Look up the things in the word_dict

		if len(seq1) > 0 and len(seq2) > 0:
			in_x1.append(seq1)
			in_x2.append(seq2)

			in_l[idx, [entity_dict[w] for w in d_words if w in entity_dict]] = 1.0
			in_y.append(entity_dict[a] if a in entity_dict else 0)


		if verbose and (idx % 10000 == 0):
			print('Vectorization: processed %d / %d' % (idx, len(examples[0])))

	def len_argsort(seq):
		return sorted(range(len(seq)), key=lambda x: len(seq[x]))

	if sort_by_len:
		sorted_index = len_argsort(in_x1)
		in_x1 = [in_x1[i] for i in sorted_index]
		in_x2 = [in_x2[i] for i in sorted_index]
		in_l = in_l[sorted_index]
		in_y = [in_y[i] for i in sorted_index]

	return in_x1, in_x2, in_l, in_y

