# coding: utf-8

import numpy as np
import numpy.linalg as ln

classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

def sigmoid(u):
  return 1. / (1. + np.e ** -u)

def back_propagation(X, labels, m, regularization=True):
  """Multilayer Neural Network
  input units:  4
  output units: 3
  :param X: d x n input matrix (d will be 4 for Iris)
  :param m: number of intermediate units
  """

  d, n = X.shape
  X = np.vstack((np.ones(n), X)).T # augumented; n x d+1

  # read label, and convert 3 unit format (001, 010, 100)
  b = -1 * np.ones((n, 3))
  for i in range(n):
    idx = classes.index(labels[i])
    b[i, idx] = 1.

  # weight matrix from input layer (d+1=3) to intermediate layer (m)
  W01 = np.random.randn(m, d+1)

  # weight matrix from intermediate layer (m) to output layer (3)
  W12 = np.random.randn(3, m)

  epoch = 0
  learning_rate = .01
  th = 1e-1
  l = .1 # lambda for regularization

  # learning
  while epoch < 10000:
    epoch += 1

    # compute output for n input data
    g10 = sigmoid(np.dot(X, W01.T)) # n x m
    g21 = sigmoid(np.dot(g10, W12.T)) # n x 3

    """ for debugging
    err = abs(b - (g21 * 2 - 1))
    print err < th, b[0], (g21 * 2 -1)[0]
    if epoch % 1000 == 0: print 'error:', sum(err), learning_rate

    # check convergence based on the threshold value
    # * NOT completely converge
    if (err < th).all(): break
    """

    # epsilon from output layer to intermediate layer
    # with converting 0, 1 output (g21) to -1, 1 output (same as b)
    e21 = ((g21 * 2 -1) - b) * g21 * (1. - g21) # n x 3

    # epsilon from intermediate layer to input layer
    e10 = np.dot(e21, W12) * g10 * (1. - g10) # n x m

    # adjust weights
    if regularization:
      W12 -= learning_rate * (np.dot(e21.T, g10) + (l * W12)) # 3 x m
      W01 -= learning_rate * (np.dot(e10.T, X) + (l * W01)) # m x d+1
    else:
      W12 -= learning_rate * np.dot(e21.T, g10) # 3 x m
      W01 -= learning_rate * np.dot(e10.T, X) # m x d+1

  return W01, W12

def cross_validation(X, labels, K, regularization=True):
  """5-fold cross validate of NN (back propagation)
  one fold has 30 tests and 120 trains
  X will be given as n x d matrix, and it will be passed to learning function as d x n matrix
  """

  total_accuracy = 0.

  idxs = range(150)
  np.random.shuffle(idxs)

  # 5-fold
  for n in range(5):
    head = 30 * n
    tail = 30 * (n + 1)
    test_idxs = idxs[head:tail]
    train_idxs = [idxs[i] for i in range(150) if (i < head or i >= tail)]

    W01, W12 = back_propagation(X[train_idxs].T, labels[train_idxs], K, regularization)

    # compute estimated class based on the result from NN
    g10 = sigmoid(np.dot(np.hstack((np.ones((30, 1)), X[test_idxs])), W01.T)) # n x m
    g21 = sigmoid(np.dot(g10, W12.T)) # n x 3

    # convert test labels -> 0, 1 arrays
    b = 0 * np.ones((30, 3))
    for i in range(30):
      idx = classes.index(labels[test_idxs[i]])
      b[i, idx] = 1.

    # compute accuracy
    result = np.round(g21)
    correct_cnt = 0
    for i in range(30):
      if (b[i] == result[i]).all(): correct_cnt += 1
    accuracy = float(correct_cnt) / 30
    total_accuracy += accuracy

  return total_accuracy / 5.

def main():
  X = []
  labels = []
  with open('iris.data') as f:
    lines = map(lambda l: l.rstrip().split(','), f.readlines())
    for line in lines:
      X.append(map(float, line[:4]))
      labels.append(line[4])

  X = np.asarray(X)
  labels = np.asarray(labels)

  print 'With regularization'
  for K in range(1, 11):
    accuracy = cross_validation(X, labels, K)
    print '[K = %2d] %f' % (K, accuracy)

  print '\nWithout regularization'
  for K in range(1, 11):
    accuracy = cross_validation(X, labels, K, False)
    print '[K = %2d] %f' % (K, accuracy)

if __name__ == '__main__':
  main()