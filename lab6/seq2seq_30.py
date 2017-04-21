from datetime import datetime
import numpy as np
import tensorflow as tf

train_length = 30
test_length = [20, 30, 50]
seq_length = 50 # Max length for training or test.
word_size = 256
batch_size = 64
hidden_size = 500
embedding_size = 100
iteration = 50000


def get_batch(length):
  batch = np.random.randint(1, word_size + 1, [length, batch_size])

  # Pad every sequence with (seq_length - length) zeros.
  batch = np.pad(batch, ((0, seq_length - length), (0, 0)), 'constant', constant_values=(0))

  feed_dict = {encoder_inputs[i] : batch[i] for i in range(seq_length)}
  feed_dict.update({targets[i] : batch[i] for i in range(seq_length)})

  return batch, feed_dict


def accuracy(sess, outputs):
  # results: shape (seq_length, batch_size, word_size + 1).
  # index: shape (seq_length, batch_size).
  acc = []
  for length in test_length:
    data, feed_dict = get_batch(length)
    results = sess.run(outputs, feed_dict=feed_dict)
    index = np.argmax(results, 2)
    acc += [np.mean(np.equal(data[:length, :], index[:length, :]).astype(float))]
  return acc


global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(1e-3, global_step, 20000, 0.9)

encoder_inputs = [tf.placeholder(tf.int32, shape=[None]) for i in range(seq_length)]

decoder_inputs = [tf.zeros_like(encoder_inputs[i]) for i in range(seq_length)]

targets = [tf.placeholder(tf.int32, shape=[None]) for i in range(seq_length)]

target_weights = [tf.placeholder(tf.float32) for i in range(seq_length)]

cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(hidden_size) for _ in range(3)])

outputs, state = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(encoder_inputs,
                                                                 decoder_inputs,
                                                                 cell,
                                                                 word_size + 1,
                                                                 word_size + 1,
                                                                 embedding_size,
                                                                 feed_previous=False)

loss = tf.contrib.legacy_seq2seq.sequence_loss(outputs, targets, target_weights)

opt = tf.train.RMSPropOptimizer(learning_rate, momentum=0.9, epsilon=1e-6)

grads_and_vars = opt.compute_gradients(loss)

clipped_grads_and_vars = [(tf.clip_by_value(grad, -9.0, 9.0), var) for grad, var in grads_and_vars]

train_op = opt.apply_gradients(clipped_grads_and_vars, global_step=global_step)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Training.
for i in range(iteration):
  _, feed_dict = get_batch(train_length)
  feed_dict.update({target_weights[i] : 1.0 for i in range(train_length)})
  feed_dict.update({target_weights[i] : 0.0 for i in range(train_length, seq_length)})
  _, l = sess.run([train_op, loss], feed_dict=feed_dict)

  if i % 100 == 0:
    acc = accuracy(sess, outputs)
    print("%s: step %d, loss = %.5f, accuracy = [%.5f, %.5f, %.5f]"
          % (datetime.now().strftime('%H:%M:%S'), i, l, acc[0], acc[1], acc[2]))

# Final test.
acc = accuracy(sess, outputs)
print("Final accuracy = [%.5f, %.5f, %.5f]" % (acc[0], acc[1], acc[2]))
