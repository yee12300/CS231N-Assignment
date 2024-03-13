from builtins import range
import numpy as np
from random import shuffle
# from past.builtins import xrange


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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    for i in range(X.shape[0]):
        unnorm_log_prob = np.dot(X[i], W)
        unnorm_prob = np.exp(unnorm_log_prob)
        norm_prob = unnorm_prob / np.sum(unnorm_prob)
        loss += -np.log(norm_prob[y[i]])
        for j in range(W.shape[1]):
            dW[:, j] += X[i] * norm_prob[j]
        dW[:, y[i]] -= X[i]

    loss /= X.shape[0]
    loss += reg * np.sum(W * W)
    dW /= X.shape[0]
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    unnorm_log_prob = np.matmul(X, W)
    unnorm_prob = np.exp(unnorm_log_prob)
    norm_prob = unnorm_prob / np.sum(unnorm_prob, axis=1, keepdims=True)
    loss = -np.sum(np.log(norm_prob[range(X.shape[0]), y])) / X.shape[0]
    loss += reg * np.sum(W * W)

    dW = np.matmul(X.T, norm_prob)
    mask = np.zeros_like(norm_prob)
    mask[range(X.shape[0]), y] = 1
    dW -= np.matmul(X.T, mask)
    dW /= X.shape[0]
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
