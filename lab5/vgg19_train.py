from six.moves import xrange
from datetime import datetime

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers

import vgg19_input

VGG_MEAN = [103.939, 116.779, 123.68] # BGR

Xtr, Ytr, Xte, Yte = vgg19_input.load_CIFAR10('cifar-10-batches-py')

x = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.int64, [None])
train_mode = tf.placeholder(tf.bool)


class Vgg19:
  def __init__(self, rand_init, use_bn=False):
    if not rand_init:
      self.data_dict = np.load('vgg19.npy', encoding='latin1').item()
    else:
      self.data_dict = None
    self.use_bn = use_bn

  def build(self, rgb, true_out, train_mode):
    # Convert RGB to BGR.
    red, green, blue = tf.split(rgb, 3, 3)
    bgr = tf.concat([blue - VGG_MEAN[0], green - VGG_MEAN[1], red - VGG_MEAN[2]], 3)

    # Do data augmentation.
    bgr = tf.cond(train_mode, lambda: self.data_augmentation(bgr), lambda: bgr)
    self.images_summary = tf.summary.image('images', bgr)

    self.conv1_1 = self.conv_layer(bgr, 3, 64, 'conv1_1')
    self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, 'conv1_2')
    self.pool1 = self.max_pool(self.conv1_2, 'pool1')

    self.conv2_1 = self.conv_layer(self.pool1, 64, 128, 'conv2_1')
    self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, 'conv2_2')
    self.pool2 = self.max_pool(self.conv2_2, 'pool2')

    self.conv3_1 = self.conv_layer(self.pool2, 128, 256, 'conv3_1')
    self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, 'conv3_2')
    self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, 'conv3_3')
    self.conv3_4 = self.conv_layer(self.conv3_3, 256, 256, 'conv3_4')
    self.pool3 = self.max_pool(self.conv3_4, 'pool3')

    self.conv4_1 = self.conv_layer(self.pool3, 256, 512, 'conv4_1')
    self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, 'conv4_2')
    self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, 'conv4_3')
    self.conv4_4 = self.conv_layer(self.conv4_3, 512, 512, 'conv4_4')
    self.pool4 = self.max_pool(self.conv4_4, 'pool4')

    self.conv5_1 = self.conv_layer(self.pool4, 512, 512, 'conv5_1')
    self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, 'conv5_2')
    self.conv5_3 = self.conv_layer(self.conv5_2, 512, 512, 'conv5_3')
    self.conv5_4 = self.conv_layer(self.conv5_3, 512, 512, 'conv5_4')

    self.fc6 = self.fc_layer(self.conv5_4, 2048, 4096, 'fc6')
    self.relu6 = tf.nn.relu(self.fc6)
    self.relu6 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu6, 0.5), lambda: self.relu6)

    self.fc7 = self.fc_layer(self.relu6, 4096, 4096, 'fc7')
    self.relu7 = tf.nn.relu(self.fc7)
    self.relu7 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu7, 0.5), lambda: self.relu7)

    self.fc8 = self.fc_layer(self.relu7, 4096, 10, 'fc8')

    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_out, logits=self.fc8))
    l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    self.loss = cross_entropy + l2 * 0.0001 # weight decay=0.0001
    self.loss_summary = tf.summary.scalar('loss', self.loss)

    correct_prediction = tf.equal(true_out, tf.argmax(self.fc8, 1))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    self.accuracy_summary = tf.summary.scalar('accuracy', self.accuracy)

  def max_pool(self, bottom, name):
    return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

  def conv_layer(self, bottom, in_channels, out_channels, name):
    with tf.variable_scope(name):
      filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)

      conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
      if self.use_bn:
        relu = tf.nn.relu(self.batch_norm(conv))
      else:
        bias = tf.nn.bias_add(conv, conv_biases)
        relu = tf.nn.relu(bias)

      return relu

  def fc_layer(self, bottom, in_size, out_size, name):
    with tf.variable_scope(name):
      weights, biases = self.get_fc_var(in_size, out_size, name)

      x = tf.reshape(bottom, [-1, in_size])
      if self.use_bn:
        fc = self.batch_norm(tf.matmul(x, weights))
      else:
        fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

      return fc

  def get_conv_var(self, filter_size, in_channels, out_channels, name):
    initial_value = tf.random_normal([filter_size, filter_size, in_channels, out_channels], stddev=0.03)
    filters = self.get_var(initial_value, name, 0)

    initial_value = tf.random_normal([out_channels], stddev=0.03)
    biases = self.get_var(initial_value, name, 1)

    return filters, biases

  def get_fc_var(self, in_size, out_size, name):
    initial_value = tf.random_normal([in_size, out_size], stddev=0.01)
    weights = self.get_var(initial_value, name, 0)

    initial_value = tf.random_normal([out_size], stddev=0.01)
    biases = self.get_var(initial_value, name, 1)

    return weights, biases

  def get_var(self, initial_value, name, idx):
    if self.data_dict is not None and name not in ('fc6', 'fc8'):
      value = self.data_dict[name][idx]
    else:
      value = initial_value

    return tf.Variable(value)

  def data_augmentation(self, bgr):
    resized_bgr = tf.map_fn(lambda image: tf.image.resize_image_with_crop_or_pad(image, 40, 40), bgr)
    cropped_bgr = tf.map_fn(lambda image: tf.random_crop(image, [32, 32, 3]), resized_bgr)
    flipped_bgr = tf.map_fn(lambda image: tf.image.random_flip_left_right(image), cropped_bgr)
    return flipped_bgr

  def batch_norm(self, bottom):
    return tf.contrib.layers.batch_norm(bottom, decay=0.9, center=True, scale=True, is_training=train_mode, updates_collections=None)


def evaluate(sess, accuracy, summary_op_eval):
  acc = 0.0
  for i in xrange(10):
    acc_, summary = sess.run([accuracy, summary_op_eval],
                             feed_dict={x: Xte[i * 1000 : (i + 1) * 1000], y: Yte[i * 1000 : (i + 1) * 1000], train_mode: False})
    acc += acc_

  return acc / 10.0, summary


def main():
  vgg = Vgg19(False, use_bn=True)
  vgg.build(x, y, train_mode)

  global_step = tf.Variable(0, trainable=False)
  boundaries = [31280, 47311]
  values = [0.01, 0.001, 0.0001]
  learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)

  train_op = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True).minimize(vgg.loss, global_step=global_step)
  lr_summary = tf.summary.scalar('learning rate', learning_rate)

  sess = tf.InteractiveSession()
  sess.run(tf.global_variables_initializer())

  summary_op_train = tf.summary.merge([vgg.images_summary, vgg.loss_summary, lr_summary])
  summary_op_eval = tf.summary.merge([vgg.accuracy_summary])
  summary_writer = tf.summary.FileWriter('vgg19_logs', sess.graph)

  f = open('result.csv', 'w')

  for step in xrange(1, 64125):
    Xtr_batch, Ytr_batch = vgg19_input.shuffle_batch(Xtr, Ytr, 128)

    _, loss_value, summary = sess.run([train_op, vgg.loss, summary_op_train],
                                      feed_dict={x: Xtr_batch, y: Ytr_batch, train_mode: True})

    if step % 100 == 0:
      summary_writer.add_summary(summary, step)
      acc, summary = evaluate(sess, vgg.accuracy, summary_op_eval)
      summary_writer.add_summary(summary, step)
      f.write(str(step * 0.001) + ', ' + str(loss_value) + ', ' + str(acc) + '\n')
      print("%s: step %d, loss = %.5f, accuracy = %.4f" % (datetime.now(), step, loss_value, acc))

  final_acc, _ = evaluate(sess, vgg.accuracy, summary_op_eval)
  print("Final accuracy = %.4f" % final_acc)


if __name__ == '__main__':
  if tf.gfile.Exists('vgg19_logs'):
    tf.gfile.DeleteRecursively('vgg19_logs')
  tf.gfile.MakeDirs('vgg19_logs')

  main()
