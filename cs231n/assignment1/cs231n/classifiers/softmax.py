import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
    # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_classes = W.shape[1]
  num_train = X.shape[0]
  
  for i in range(num_train):
      scores = X[i].dot(W)
      correct_class = y[i]
      exp_scores = np.zeros_like(scores)
      row_sum = 0
      for j in range(num_classes):
          exp_scores[j] = np.exp(scores[j])
          row_sum += exp_scores[j]
      loss += -np.log(exp_scores[correct_class]/row_sum)
      #compute dW loops:
      for k in range(num_classes):
        if k != correct_class:
          dW[:,k] += exp_scores[k] / row_sum * X[i]
        else:
          dW[:,k] += (exp_scores[correct_class]/row_sum - 1) * X[i]
  loss /= num_train
  reg_loss = 0.5 * reg * np.sum(W**2)
  loss += reg_loss
  dW /= num_train
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  print(X)
  num_train = X.shape[0]
  scores = X.dot(W)
  correct_class_score = scores[np.arange(num_train),y].reshape(num_train,1)
  exp_sum = np.sum(np.exp(scores),axis=1).reshape(num_train,1)
  loss += np.sum(np.log(exp_sum) - correct_class_score)
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)

  #compute gradient
  margin = np.exp(scores) / exp_sum
  margin[np.arange(num_train),y] += -1
  dW = X.T.dot(margin)
  dW /= num_train
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

