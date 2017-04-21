from six.moves import xrange

import _pickle
import numpy as np
import os
import random


""" Load the CIFAR-10 dataset. """

def load_CIFAR10_batch(filename):
  with open(filename, 'rb') as f:
    datadict = _pickle.load(f, encoding='latin1')
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32)
    Y = np.array(Y).astype(np.int64)
    return X, Y


def load_CIFAR10(data_dir):
  xs = []
  ys = []
  for b in xrange(1, 6):
    f = os.path.join(data_dir, 'data_batch_%d' % b)
    X, Y = load_CIFAR10_batch(f)
    xs.append(X)
    ys.append(Y)
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  Xte, Yte = load_CIFAR10_batch(os.path.join(data_dir, 'test_batch'))
  return Xtr, Ytr, Xte, Yte


def shuffle_batch(X, Y, batch_size):
  data_idx = random.sample(xrange(50000), batch_size)
  return X[data_idx], Y[data_idx]
