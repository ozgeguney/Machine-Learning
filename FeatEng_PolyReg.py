import numpy as np
import matplotlib.pyplot as plt
from BasicFunctions import zscore_normalize_features, run_gradient_descent, compute_cost_matrix

x = np.arange(0,20,1)
y = x**2

X = np.c_[x,x**2,x**3]

w, b = run_gradient_descent(X,y,10000,0.0000001)
y_predicted = X @ w + b

plt.scatter(x,y, c="r", marker="x", label="Actual values")
plt.title("Feature Engineering")
plt.plot(x, y_predicted, label = "Predicted Value")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

X_features = ["x","x^2","x^3"]
fig,ax = plt.subplots(1,3,figsize=(12,3), sharey= True)
for i in range(len(ax)):
    ax[i].scatter(X[:,i],y)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("y")


X = zscore_normalize_features(X)
w, b = run_gradient_descent(X,y,10000,0.1)
compute_cost_matrix(X,y,w,b)
y_predicted = X @ w + b
fig1 = plt.figure("Figure 3")
plt.scatter(x, y, marker="o", c= "y",label= "Actual Value")
plt.title("Normalized with Feature Scaling")
plt.plot(x,y_predicted,label="Predicted Value")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()


