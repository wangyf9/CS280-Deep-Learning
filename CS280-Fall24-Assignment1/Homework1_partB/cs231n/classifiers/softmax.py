from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

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

    N = X.shape[0]
    C = W.shape[1]
    
    for i in range(N):
        scores = np.dot(X[i], W)
        label = y[i]
        # Shift score to increase the data stablity
        shiftscore = scores - np.max(scores)
        total_exp = np.sum(np.exp(shiftscore))
        loss_i = - shiftscore[label] + np.log(total_exp)
        loss += loss_i
        for j in range(C):
            exp_score = np.exp(shiftscore[j])/ total_exp       
            if j == label:
                dW[:, j] += (-1 + exp_score) * X[i]
            else:
                dW[:, j] += exp_score * X[i]
    # regularization + average
    loss /= N
    loss += 0.5* reg* np.sum(W*W)
    dW = dW/ N + reg* W
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
    N = X.shape[0]
    C = W.shape[1]
    scores = np.dot(X, W)
    shiftscores = scores - np.max(scores, axis = 1).reshape(-1, 1)
    exp_score = np.exp(shiftscores)/np.sum(np.exp(shiftscores), axis = 1).reshape(-1, 1)
    loss = -np.sum(np.log(exp_score[range(N), list(y)]))
    loss /= N
    loss += 0.5* reg *np.sum(W*W)

    dS = exp_score.copy()
    dS[range(N), list(y)] += -1
    dW = np.dot((X.T), dS)
    dW = dW/N + reg* W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
