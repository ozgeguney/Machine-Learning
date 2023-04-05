import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
def load_house_data():
    data = np.loadtxt("/Users/ozgeguney/PycharmProjects/pythonProject/data/houses.txt", delimiter=',', skiprows=1)
    X = data[:,:4]
    y = data[:,4]
    return X, y

X_train, y_train = load_house_data()
X_features = ['size(sqft)','bedrooms','floors','age']

fig,ax = plt.subplots(1,4, figsize=(12,3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i], y_train)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("Price")
#plt.show()

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
    w_out, b_out = gradient_descent_houses(x,y,w_initial,b_initial, alpha, iteration, compute_gradient )
    #print(f"w,b found by gradient descent: w: {w_out}, b: {b_out:0.2f}")
    print(w_out,b_out)
    return (w_out, b_out)

import copy

def gradient_descent_houses(X,y,w,b,alpha,iteration,gradient_function):
    m = len(X)
    w_temp = copy.deepcopy(w)
    b_temp = b
    for i in range(iteration):
        djw,djb =gradient_function(X,y,w_temp,b_temp)
        w_temp = w_temp - alpha*djw
        b_temp = b_temp - alpha*djb
    return w_temp, b_temp



w_out, b_out= run_gradient_descent(X_train,y_train,100,0.0000001)
compute_cost(X_train,y_train,w_out,b_out)
compute_cost_matrix(X_train,y_train,w_out,b_out)

def zscore_normalize_features(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return (X_norm)

X_norm= zscore_normalize_features(X_train)
fig,ax=plt.subplots(1, 4, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_norm[:,i],y_train)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("Price (1000's)")
#plt.show()

#The scaled features get very accurate results much, much faster!
#Notice the gradient of each parameter is tiny by the end of this fairly short run.
w_out, b_out= run_gradient_descent(X_norm,y_train,100,0.1)
compute_cost(X_norm,y_train,w_out,b_out)


