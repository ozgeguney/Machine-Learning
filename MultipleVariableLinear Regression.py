import copy

import numpy as np
import matplotlib.pyplot as plt

#The training dataset(matrix) contains three examples with four features m=3 , n=4
X_train = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
y_train = np.array([10, 20, 30])

b_init = 0
# n is 4, w is a 1-D NumPy vector.
w_init = np.array([1, 4, -4, 2])
def predict_loop(x, w, b):
    m = x.shape[0]
    n = x.shape[1]
    f_wb =[]
    for j in range(m):
        predict = 0
        for i in range(n):
            predict += w[i]*x[j,i]
        predict += b
        f_wb.append(predict)
    return f_wb

f_wb = predict_loop(X_train, w_init, b_init)
print(f_wb)

def predict(x,w,b):
    f_wb = np.dot(x,w) +b
    return f_wb

print(predict(X_train,w_init,b_init))

def compute_cost(x,y,w,b):
    m = x.shape[0]
    cost =0
    prediction = []
    for i in range(m):
        predict = np.dot(x[i],w) + b
        prediction.append(predict)
        cost += (y[i] - predict)**2
    cost = cost / (2*m)
    return cost, prediction

cost, prediction = compute_cost(X_train,y_train,w_init,b_init)
print(cost, prediction)

def compute_gradient(x,y,w,b):
    m,n= x.shape
    dj_w = np.zeros((n,))
    dj_b = 0
    for i in range(m):
        predict = np.dot(x[i],w) +b
        err = predict -y[i]
        for j in range(n):
            dj_w[j] = err*x[i,j]
        dj_b += err
    dj_w = dj_w/m
    dj_b = dj_b/m
    return  dj_w,dj_b

def compute_gradient_descent(x,y,w,b, alpha,iteration, gradient_function):
    w_temp = copy.deepcopy(w)
    b_temp = b
    for i in range(iteration):
        dj_w, dj_b = gradient_function(x,y,w_temp,b_temp)
        w_temp = w_temp - alpha*dj_w
        b_temp = b_temp - alpha*dj_b
    return w_temp,b_temp

print(compute_gradient_descent(X_train,y_train,w_init,b_init,0.01,9000, compute_gradient))
a = (compute_cost(X_train,y_train,[0.89285714,  3.88095238, -4.13095238,  1.85714286], 6.309523809523661)[0])
print(f"Cost {a:8.2f}   ")
print(compute_cost(X_train,y_train,w_init, b_init))






