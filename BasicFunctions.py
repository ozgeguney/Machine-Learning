import numpy as np


def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost =0
    for i in range(m):
        fx= np.dot(X[i],w) + b
        cost += (fx-y[i])**2
    cost = cost/(2*m) #np.squeeze(cost)
    print(cost)
    return cost
def compute_cost_matrix(X, y, w, b):
    fx = X @ w + b
    err = np.sum((fx - y)**2)/(2*X.shape[0])
    print(err)
    return err

def compute_gradient(X,y,w,b):
    m,n = X.shape
    djw = np.zeros(n)
    djb = 0
    for i in range(m):
        fx = np.dot(X[i],w) + b
        err = fx - y[i]
        for j in range(n):
            djw[j] += err * X[i,j]
        djb += err
    djw = djw / m
    djb = djb / m
    return djw, djb

def compute_gradient_matrix(X, y, w, b):
    m = X.shape[0]
    # calculate fx for all examples.
    fx = X @ w + b # x: m,n w: n, fx: m
    err = fx - y
    djw = (X.T @ err)/m
    djb = np.sum(err)/ m
    return djw,djb

def run_gradient_descent(x,y,iteration,alpha):
    w_initial = np.zeros(x.shape[1])
    b_initial =0
    w_out, b_out = gradient_descent(x,y,w_initial,b_initial, alpha, iteration, compute_gradient )
    print(f"w,b found by gradient descent: w: {w_out}, b: {b_out:0.4f}")
    #print(w_out,b_out)
    return (w_out, b_out)

import copy

def gradient_descent(X, y, w, b, alpha, num_iters, gradient_function):
    m = len(X)
    w_temp = copy.deepcopy(w)
    b_temp = b
    for i in range(num_iters):
        djw, djb = gradient_function(X, y, w_temp, b_temp)
        w_temp = w_temp - alpha * djw
        b_temp = b_temp - alpha * djb
    return w_temp, b_temp

def zscore_normalize_features(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return (X_norm)