import numpy as np
from BasicFunctions import compute_cost,run_gradient_descent, compute_cost_matrix, zscore_normalize_features
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
plt.show()

w_out, b_out= run_gradient_descent(X_train,y_train,100,0.0000001)
#compute_cost(X_train,y_train,w_out,b_out)
print("Without normalization: ")
compute_cost_matrix(X_train,y_train,w_out,b_out)

X_norm= zscore_normalize_features(X_train)
fig,ax=plt.subplots(1, 4, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_norm[:,i],y_train)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("Price (1000's)")
plt.show()

#The scaled features get very accurate results much, much faster!
#Notice the gradient of each parameter is tiny by the end of this fairly short run.
w_out, b_out= run_gradient_descent(X_norm,y_train,100,0.1)
print("With normalization: ")
compute_cost(X_norm,y_train,w_out,b_out)


