from scipy import misc

import numpy as np
import math
import tensorflow as tf

VGG_MEAN = [103.939, 116.779, 123.68] # BGR


class Vgg19:
  def __init__(self):
    self.data_dict = np.load('vgg19.npy', encoding='latin1').item()
    print("npy file loaded")

  def build(self, rgb):
    print("build model started")

    # Convert RGB to BGR.
    rgb = tf.cast(rgb, tf.float32)
    red, green, blue = tf.split(rgb, 3, 3)
    bgr = tf.concat([blue - VGG_MEAN[0], green - VGG_MEAN[1], red - VGG_MEAN[2]], 3)

    self.conv1_1 = self.conv_layer(bgr, 'conv1_1')
    self.conv1_2 = self.conv_layer(self.conv1_1, 'conv1_2')
    self.pool1 = self.max_pool(self.conv1_2, 'pool1')

    self.conv2_1 = self.conv_layer(self.pool1, 'conv2_1')
    self.conv2_2 = self.conv_layer(self.conv2_1, 'conv2_2')
    self.pool2 = self.max_pool(self.conv2_2, 'pool2')

    self.conv3_1 = self.conv_layer(self.pool2, 'conv3_1')
    self.conv3_2 = self.conv_layer(self.conv3_1, 'conv3_2')
    self.conv3_3 = self.conv_layer(self.conv3_2, 'conv3_3')
    self.conv3_4 = self.conv_layer(self.conv3_3, 'conv3_4')
    self.pool3 = self.max_pool(self.conv3_4, 'pool3')

    self.conv4_1 = self.conv_layer(self.pool3, 'conv4_1')
    self.conv4_2 = self.conv_layer(self.conv4_1, 'conv4_2')
    self.conv4_3 = self.conv_layer(self.conv4_2, 'conv4_3')
    self.conv4_4 = self.conv_layer(self.conv4_3, 'conv4_4')
    self.pool4 = self.max_pool(self.conv4_4, 'pool4')

    self.conv5_1 = self.conv_layer(self.pool4, 'conv5_1')
    self.conv5_2 = self.conv_layer(self.conv5_1, 'conv5_2')
    self.conv5_3 = self.conv_layer(self.conv5_2, 'conv5_3')
    self.conv5_4 = self.conv_layer(self.conv5_3, 'conv5_4')
    self.pool5 = self.max_pool(self.conv5_4, 'pool5')

    self.fc6 = self.fc_layer(self.pool5, 'fc6')
    self.relu6 = tf.nn.relu(self.fc6)

    self.fc7 = self.fc_layer(self.relu6, 'fc7')
    self.relu7 = tf.nn.relu(self.fc7)

    self.fc8 = self.fc_layer(self.relu7, 'fc8')

    self.prob = tf.nn.softmax(self.fc8, name='prob')

    print("build model finished")

  def max_pool(self, bottom, name):
    return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

  def conv_layer(self, bottom, name):
    with tf.variable_scope(name):
      filt = self.get_conv_filter(name)
      conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

      conv_biases = self.get_bias(name)
      bias = tf.nn.bias_add(conv, conv_biases)

      relu = tf.nn.relu(bias)
      return relu

  def fc_layer(self, bottom, name):
    with tf.variable_scope(name):
      shape = bottom.get_shape().as_list()
      dim = np.prod(shape[1:])
      x = tf.reshape(bottom, [-1, dim])

      weights = self.get_fc_weight(name)
      biases = self.get_bias(name)

      fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

      return fc

  def get_conv_filter(self, name):
    return tf.constant(self.data_dict[name][0], name='filters')

  def get_bias(self, name):
    return tf.constant(self.data_dict[name][1], name='biases')

  def get_fc_weight(self, name):
    return tf.constant(self.data_dict[name][0], name='weights')


def preprocess(img):
  smallest_side = min(img.shape[:2])
  fraction = 224 / smallest_side
  new_shape = [math.ceil(x * fraction) for x in img.shape[:2]]
  resized_img = misc.imresize(img, new_shape)

  # Use a center crop.
  yy = (resized_img.shape[0] - 224) // 2
  xx = (resized_img.shape[1] - 224) // 2
  cropped_img = resized_img[yy:yy+224, xx:xx+224]

  return cropped_img


def print_top5(prob):
  synset = [l.strip().partition(' ')[2] for l in open('synset.txt').readlines()]

  # Sort prob from large to small.
  pred = np.argsort(prob)[::-1]

  for i in range(5): print(prob[pred[i]], synset[pred[i]])


if __name__ == '__main__':
  image = misc.imread('test_data/tiger.jpeg', mode='RGB')
  image = preprocess(image)
  image = image.reshape(1, 224, 224, 3)

  images = tf.placeholder(tf.int32, [None, 224, 224, 3])
  sess = tf.Session()

  vgg = Vgg19()
  vgg.build(images)

  prob = sess.run(vgg.prob, feed_dict={images:image})
  print_top5(prob[0])
