import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_train = np.array([250, 300, 480,  430,   630, 730,])

#try to find optimal values. it will minimize cost.
w = 210
b = 2

def compute_cost(x_train, y_train, w, b):
    m = x_train.shape[0]
    prediction = np.zeros(m)
    cost =0
    #model f(x) = wx + b
    for i in range(m):
        prediction[i] = w*x_train[i] + b
        cost += (prediction[i] - y_train[i])**2
    averageCost = cost / (2*m)
    return averageCost, prediction

cost, prediction = compute_cost(x_train,y_train,w,b)

print(cost)
plt.plot(x_train, prediction, c="b", label = "Our Prediction")
plt.scatter(x_train, y_train, c="r", marker="x", label = "Actual Values")
plt.title("House Pricing")
plt.xlabel("Size")
plt.ylabel("Price")
plt.legend()
plt.show()

