import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def load_house_data():
    data = np.loadtxt("/Users/ozgeguney/PycharmProjects/pythonProject/data/houses.txt", delimiter=',', skiprows=1)
    X = data[:,:4]
    y = data[:,4]
    return X, y

X_train, y_train = load_house_data()
X_features = ["size", "bedrooms", "floors", "age"]

scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train)

from sklearn.linear_model import SGDRegressor
sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(X_norm,y_train)
#print(f"number of iterations completed: {sgdr.n_iter_}, number of weight updates: {sgdr.t_}")
b = sgdr.intercept_
w = sgdr.coef_
print(f"sgdr:  w: {w}, b:{b}")
y_predict_sgdr = sgdr.predict(X_norm)
#y_predict = X_norm @ w + b # np.dot(X_norm, w) + b
print(f"sgdr : Prediction on training set:\n{y_predict_sgdr[:4]}" )
#print(f"Target values \n{y_train[:4]}")

#The closed-form solution does not require normalization.
#The closed-form solution work well on smaller data sets such as these
#but can be computationally demanding on larger data sets.
from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
linear_model.fit(X_train,y_train)
w2 = linear_model.coef_
b2 = linear_model.intercept_
print(f"linear_model w: {w2:}, b: {b2:0.2f}")
print(f"linear_model : Prediction on training set:\n {linear_model.predict(X_train)[:4]}" )
#print(f"prediction using w,b:\n {(X_train @ w2 + b2)[:4]}")
print(f"Target values \n {y_train[:4]}")

fig, ax = plt.subplots(1,4,figsize=(12,3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train,label="target")
    ax[i].set_xlabel(X_features[i])
    ax[i].scatter(X_train[:,i], y_predict_sgdr, color="r",label="predict")
ax[0].set_ylabel("Price")
ax[0].legend()
fig.suptitle("target versus prediction using z-score normalized model")
plt.show()

x_house = np.array([1200, 3,1, 40]).reshape(-1,4)
a= linear_model.predict(x_house)
x_house_predict = linear_model.predict(x_house)[0]
print(f" predicted price = {x_house_predict*1000:0.2f}")

