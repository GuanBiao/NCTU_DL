from six.moves import xrange
from datetime import datetime

import tensorflow as tf

import cifar10_input

Xtr, Ytr, Xte, Yte = cifar10_input.load_CIFAR10('cifar-10-batches-py')
Xtr, Xte = cifar10_input.data_preprocessing(Xtr, Xte)

x = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int64, [None])
keep_prob = tf.placeholder(tf.float32)
distort_data = tf.placeholder(tf.bool) # indicate whether to distort data for data augmentation

global_step = tf.Variable(0, trainable=False)

# Use a learning rate that is 0.1 for the first 80 epochs, 0.01 for epochs 81~121,
# and 0.001 for any additional epochs.
boundaries = [31280, 47311]
values = [0.1, 0.01, 0.001]
learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)


""" Do data augmentation. """

def data_augmentation():
  resized_x = tf.map_fn(lambda image: tf.image.resize_image_with_crop_or_pad(image, 40, 40), x)
  cropped_x = tf.map_fn(lambda image: tf.random_crop(image, [32, 32, 3]), resized_x)
  flipped_x = tf.map_fn(lambda image: tf.image.random_flip_left_right(image), cropped_x)
  return flipped_x


""" Build the NIN model. """

def weight_variable(shape, stddev=0.05):
  initial = tf.random_normal(shape, stddev=stddev)
  return tf.Variable(initial)


def bias_variable(shape, stddev=0.05):
  initial = tf.random_normal(shape, stddev=stddev)
  return tf.Variable(initial)


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_3x3(x):
  return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')


def avg_pool_8x8(x):
  return tf.nn.avg_pool(x, ksize=[1, 8, 8, 1], strides=[1, 1, 1, 1], padding='VALID')


def inference(x):
  # conv1
  W_conv1 = weight_variable([5, 5, 3, 192], 0.01)              # filter = 5x5, in = 3, out = 192
  b_conv1 = bias_variable([192], 0.01)
  h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)           # out = 32x32x192

  # mlp1-1
  W_mlp1_1 = weight_variable([1, 1, 192, 160])                 # filter = 1x1, in = 192, out = 160
  b_mlp1_1 = bias_variable([160])
  h_mlp1_1 = tf.nn.relu(conv2d(h_conv1, W_mlp1_1) + b_mlp1_1)  # out = 32x32x160

  # mlp1-2
  W_mlp1_2 = weight_variable([1, 1, 160, 96])                  # filter = 1x1, in = 160, out = 96
  b_mlp1_2 = bias_variable([96])
  h_mlp1_2 = tf.nn.relu(conv2d(h_mlp1_1, W_mlp1_2) + b_mlp1_2) # out = 32x32x96

  # pool, dropout
  h_pool = max_pool_3x3(h_mlp1_2)                              # out = 16x16x96
  h_drop = tf.nn.dropout(h_pool, keep_prob)

  # conv2
  W_conv2 = weight_variable([5, 5, 96, 192])                   # filter = 5x5, in = 96, out = 192
  b_conv2 = bias_variable([192])
  h_conv2 = tf.nn.relu(conv2d(h_drop, W_conv2) + b_conv2)      # out = 16x16x192

  # mlp2-1
  W_mlp2_1 = weight_variable([1, 1, 192, 192])                 # filter = 1x1, in = 192, out = 192
  b_mlp2_1 = bias_variable([192])
  h_mlp2_1 = tf.nn.relu(conv2d(h_conv2, W_mlp2_1) + b_mlp2_1)  # out = 16x16x192

  # mlp2-2
  W_mlp2_2 = weight_variable([1, 1, 192, 192])                 # filter = 1x1, in = 192, out = 192
  b_mlp2_2 = bias_variable([192])
  h_mlp2_2 = tf.nn.relu(conv2d(h_mlp2_1, W_mlp2_2) + b_mlp2_2) # out = 16x16x192

  # pool, dropout
  h_pool = max_pool_3x3(h_mlp2_2)                              # out = 8x8x192
  h_drop = tf.nn.dropout(h_pool, keep_prob)

  # conv3
  W_conv3 = weight_variable([3, 3, 192, 192])                  # filter = 3x3, in = 192, out = 192
  b_conv3 = bias_variable([192])
  h_conv3 = tf.nn.relu(conv2d(h_drop, W_conv3) + b_conv3)      # out = 8x8x192

  # mlp3-1
  W_mlp3_1 = weight_variable([1, 1, 192, 192])                 # filter = 1x1, in = 192, out = 192
  b_mlp3_1 = bias_variable([192])
  h_mlp3_1 = tf.nn.relu(conv2d(h_conv3, W_mlp3_1) + b_mlp3_1)  # out = 8x8x192

  # mlp3-2
  W_mlp3_2 = weight_variable([1, 1, 192, 10])                  # filter = 1x1, in = 192, out = 10
  b_mlp3_2 = bias_variable([10])
  h_mlp3_2 = tf.nn.relu(conv2d(h_mlp3_1, W_mlp3_2) + b_mlp3_2) # out = 8x8x10

  # pool
  h_pool = avg_pool_8x8(h_mlp3_2)                              # out = 1x10
  output = tf.reshape(h_pool, [-1, 10])

  return output


""" Test the NIN model. """

def evaluate(sess, accuracy, summary_op_eval):
  acc = 0.0
  for i in xrange(10):
    acc_, summary = sess.run([accuracy, summary_op_eval],
                             feed_dict={x: Xte[i * 1000 : (i + 1) * 1000],
                                        y: Yte[i * 1000 : (i + 1) * 1000],
                                        keep_prob: 1.0, distort_data: False})
    acc += acc_

  return acc / 10.0, summary


""" Train and test the NIN model on the CIFAR-10 dataset. """

def main():
  # Do data augmentation.
  distorted_x = tf.cond(distort_data, data_augmentation, lambda: x)
  images_summary = tf.summary.image('images', distorted_x)

  # Build the NIN model.
  logits = inference(distorted_x)

  # Add loss.
  cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                                                logits=logits))
  l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
  loss = cross_entropy + l2 * 0.0001 # weight decay=0.0001
  loss_summary = tf.summary.scalar('loss', loss)

  # Train the NIN model.
  train_op = tf.train.MomentumOptimizer(learning_rate, 0.9,
                                        use_nesterov=True).minimize(loss, global_step=global_step)
  lr_summary = tf.summary.scalar('learning rate', learning_rate)

  # Test the NIN model.
  correct_prediction = tf.equal(y, tf.argmax(logits, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  accuracy_summary = tf.summary.scalar('accuracy', accuracy)

  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())

  # Merge summaries and write them out to cifar10_logs.
  summary_op_train = tf.summary.merge([images_summary, loss_summary, lr_summary])
  summary_op_eval = tf.summary.merge([accuracy_summary])
  summary_writer = tf.summary.FileWriter('cifar10_logs', sess.graph)

  f = open('result.csv', 'w')

  # 164 epochs, each of which has 391 iterations.
  for step in xrange(1, 64125):
    Xtr_batch, Ytr_batch = cifar10_input.shuffle_batch(Xtr, Ytr, 128)

    _, loss_value, summary = sess.run([train_op, loss, summary_op_train],
                                      feed_dict={x: Xtr_batch, y: Ytr_batch,
                                                 keep_prob: 0.5, distort_data: True})

    # Record the processed images, the loss for a training batch, the learning rate,
    # and the accuracy for a test batch every 100 steps.
    if step % 100 == 0:
      summary_writer.add_summary(summary, step)
      acc, summary = evaluate(sess, accuracy, summary_op_eval)
      summary_writer.add_summary(summary, step)
      f.write(str(loss_value) + ', ' + str(acc) + '\n')
      print("%s: step %d, loss = %.5f, accuracy = %.4f" % (datetime.now(), step, loss_value, acc))

  final_acc, _ = evaluate(sess, accuracy, summary_op_eval)
  print("Final accuracy = %.4f" % final_acc)


if __name__ == '__main__':
  if tf.gfile.Exists('cifar10_logs'):
    tf.gfile.DeleteRecursively('cifar10_logs')
  tf.gfile.MakeDirs('cifar10_logs')

  main()
