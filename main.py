import utils
import logging
import pdb
import config
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
# import tensorflow.compat.v1 as tf


def gen_examples(x1, x2, l, y, batch_size):
	minibatches = utils.get_minibatches(len(x1), batch_size)
	all_ex = []

	for minibatch in minibatches:

		mb_x1 = [x1[t] for t in minibatch]
		mb_x2 = [x2[t] for t in minibatch]
		mb_l = l[minibatch]
		mb_y = [y[t] for t in minibatch]

		mb_x1, mb_mask1 = utils.prepare_data(mb_x1)
		mb_x2, mb_mask2 = utils.prepare_data(mb_x2)

		all_ex.append((mb_x1, mb_mask1, mb_x2, mb_mask2, mb_l, mb_y))

	return all_ex


num_epochs = 10000
embedding_size = 100
embedding_file = 'data/glove.6B/glove.6B.50d.txt'
hidden_size = 128
embedding_file = None
dropout_rate = 0.2
learning_rate=0.05
eval_iter = 10
batch_size = 10


file_name = '/Users/yangsun/Desktop/dataset/training_cnn.txt'
val_file_name = '/Users/yangsun/Desktop/dataset/validation_cnn.txt'
model_path = './model_path'

documents, questions, answers = utils.load_data(file_name, 10)
word_dict = utils.build_dict(documents + questions)

documents_val, questions_val, answers_val = utils.load_data(val_file_name, 100)
word_dict_val = utils.build_dict(documents_val + questions_val)

entity_markers = list(set([w for w in word_dict.keys() if w.startswith('@entity')] + answers))


entity_markers = ['<unk_entity>'] + entity_markers
entity_dict = {w: index for (index, w) in enumerate(entity_markers)}
num_labels = len(entity_dict)
embeddings = utils.gen_embeddings(word_dict, embedding_size, embedding_file)
vocab_size, embedding_size = embeddings.shape


# tf.reset_default_graph()
d_input = tf.placeholder(dtype=tf.int32, shape=(None, None), name="d_input")
q_input = tf.placeholder(dtype=tf.int32, shape=(None, None), name="q_input") # [batch_size, max_seq_length_for_batch]
l_mask = tf.placeholder(dtype=tf.float32, shape=(None, None), name="l_mask") # [batch_size, entity num]


y = tf.placeholder(dtype=tf.int32, shape=None, name="label") # batch size vector
y_1hot= tf.placeholder(dtype=tf.float32, shape=(None, None), name="label_1hot") # onehot encoding of y [batch_size, entitydict]
training = tf.placeholder(dtype=tf.bool)


word_embeddings = tf.get_variable("glove", shape=(vocab_size, embedding_size), initializer=tf.constant_initializer(embeddings))
W_bilinear = tf.Variable(tf.random_uniform((2*hidden_size, 2*hidden_size), minval=-0.01, maxval=0.01))


with tf.variable_scope('d_encoder'): # Encoding Step for Passage (d_ for document)
	d_embed = tf.nn.embedding_lookup(word_embeddings, d_input) # Apply embeddings: [batch, max passage length in batch, GloVe Dim]
	d_embed_dropout = tf.layers.dropout(d_embed, rate=dropout_rate, training=training) # Apply Dropout to embedding layer

	d_cell_fw = rnn.GRUCell(hidden_size) # TODO: kernel_initializer=tf.random_normal_initializer(0,0.1) not working for 1.1
	d_cell_bw = rnn.GRUCell(hidden_size)

	d_outputs, _ = tf.nn.bidirectional_dynamic_rnn(d_cell_fw, d_cell_bw, d_embed_dropout, dtype=tf.float32)
	d_output = tf.concat(d_outputs, axis=-1) # [batch, len, h], len is the max passage length, and h is the hidden size
	# shape=(?, ?, 128)

with tf.variable_scope('q_encoder'):
	q_embed = tf.nn.embedding_lookup(word_embeddings, q_input)
	q_embed_dropout = tf.layers.dropout(q_embed, rate = dropout_rate, training=training)

	q_cell_fw = rnn.GRUCell(hidden_size)
	q_cell_bw = rnn.GRUCell(hidden_size)

	q_outputs, q_laststates = tf.nn.bidirectional_dynamic_rnn(q_cell_fw, q_cell_bw, q_embed_dropout, dtype=tf.float32)
	q_output = tf.concat(q_laststates, axis=-1) # (batch, h)


with tf.variable_scope('bilinear'):
	M = d_output * tf.expand_dims(tf.matmul(q_output, W_bilinear), axis=1)
	alpha = tf.nn.softmax(tf.reduce_sum(M, axis=2)) # [batch, len]
	# this output contains the weighted combination of all contextual embeddings
	bilinear_output = tf.reduce_sum(d_output * tf.expand_dims(alpha, axis=2), axis=1) # [batch, h]


with tf.variable_scope('dense'): # Prediction Step
	# the final output has dimension [batch, entity#], giving the probabilities of an entity being the answer for examples
	final_prob = tf.layers.dense(bilinear_output, units=num_labels, activation=tf.nn.softmax, kernel_initializer=tf.random_uniform_initializer(minval=-0.01, maxval=0.01)) # [batch, entity#]



pred = final_prob * l_mask # ignore entities that don't appear in the passage
train_pred = pred / tf.expand_dims(tf.reduce_sum(pred, axis=1), axis=1) # redistribute probabilities ignoring certain labels
train_pred = tf.clip_by_value(train_pred, 1e-7, 1.0 - 1e-7)


test_pred = tf.cast(tf.argmax(pred, axis=-1), tf.int32)
acc = tf.reduce_sum(tf.cast(tf.equal(test_pred, y), tf.int32))


loss_op = tf.reduce_mean(-tf.reduce_sum(y_1hot * tf.log(train_pred), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)


dev_x1, dev_x2, dev_l, dev_y = utils.vectorize((documents, questions, answers), word_dict, entity_dict)
all_train = gen_examples(dev_x1, dev_x2, dev_l, dev_y, batch_size)


val_x1, val_x2, val_l, val_y = utils.vectorize((documents_val, questions_val, answers_val), word_dict_val, entity_dict)
all_val = gen_examples(val_x1, val_x2, val_l, val_y, batch_size)


init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_updates = 0
best_acc = 0.0

with tf.Session() as sess:
	sess.run(init)

	for e in range(num_epochs):
		np.random.shuffle(all_train)

		for idx, (mb_x1, mb_mask1, mb_x2, mb_mask2, mb_l, mb_y) in enumerate(all_train):
			print('Batch Size = %d, # of Examples = %d, max_len = %d' % (mb_x1.shape[0], len(mb_x1), mb_x1.shape[1]))

			y_label = np.zeros((mb_x1.shape[0], num_labels))

			for r, i in enumerate(mb_y):
				y_label[r][i] = 1

			_, train_loss = sess.run([train_op, loss_op], feed_dict={d_input:mb_x1, q_input:mb_x2, y_1hot: y_label, l_mask: mb_l, training: True})
			pred_ = sess.run(train_pred, feed_dict={d_input:mb_x1, q_input:mb_x2, y_1hot: y_label, l_mask: mb_l, training: True})

			print('Epoch = %d, Iter = %d (max = %d), Loss = %.2f,' %
			                (e, idx, len(all_train), train_loss))
			n_updates += 1

			if n_updates % eval_iter != 0:
				continue

			saver.save(sess, model_path, global_step=e)
			correct = 0
			n_examples = 0
			for d_x1, d_mask1, d_x2, d_mask2, d_l, d_y in all_train:
				correct += sess.run(acc, feed_dict = {d_input:d_x1, q_input:d_x2, y: d_y, l_mask: d_l, training: False})
				n_examples += len(d_x1)
			dev_acc = correct * 100. / n_examples
			print('Dev Accuracy: %.2f %%' % dev_acc)
			if dev_acc > best_acc:
				best_acc = dev_acc
				print('Best Dev Accuracy: epoch = %d, n_updates (iter) = %d, acc = %.2f %%' %
		                        (e, n_updates, dev_acc))
