import numpy as np

x_train =np.array([1.0,2.0])
y_train =np.array([300.0, 500.0])

def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dCost_w = 0
    dCost_b = 0
    for i in range(m):
        prediction = w* x[i] + b
        dCost_w += (prediction - y[i]) * x[i]
        dCost_b += prediction - y[i]
    dCost_w = dCost_w / m
    dCost_b = dCost_b / m
    return dCost_w, dCost_b

def gradient_descent(x,y, w_initial, b_initial, alpha, num_iters, gradient_function):
    w = w_initial
    b = b_initial

    w_b_history =[]
    for i in range(num_iters):
        dCost_w, dCost_b = gradient_function(x,y,w,b)
        w = w - alpha * dCost_w
        b = b - alpha * dCost_b

        w_b_history.append([w,b])

    return w,b, w_b_history

w_init = 0
b_init = 0
alpha = 0.5
iterations = 400

w_final, b_final, w_b_history = gradient_descent(x_train,y_train,w_init,b_init,alpha,iterations, compute_gradient)
print(f"(w,b) found by gradient descent: ({w_final:0.4f},{b_final:0.4f})")
print(w_b_history)



