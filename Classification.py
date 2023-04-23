import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([0.,1,2,3,4,5])
y_train = np.array([0,0,0,1,1,1])

x_train2 = np.array([[0.5,1.5],[1,1],[1.5,0.5],[3,0.5],[2,2],[1,2.5]])
y_train2 = np.array([0,0,0,1,1,1])

positive = y_train==1
negative = y_train==0

fig,ax = plt.subplots(1,2, figsize = (12,5))
ax[0].scatter(x_train[positive], y_train[positive], marker="x", s=80, c="r", label="positive")
ax[0].scatter(x_train[negative], y_train[negative], marker="o", s=100, c="b", label="negative")

ax[0].set_title("one variable plot")
ax[0].set_xlabel("x", fontsize =12)
ax[0].set_ylabel("y", fontsize =12)
ax[0].legend()

ax[1].scatter(x_train2[y_train2==1][:,0], x_train2[y_train2==1][0:,1], marker="x", s=80, c="r", label="positive")
ax[1].scatter(x_train2[y_train2==0][:,0], x_train2[y_train2==0][0:,1], marker="o", s=100, c="b", label="negative")
ax[1].set_ylabel('x_1', fontsize=12)
ax[1].set_xlabel('x_0', fontsize=12)
ax[1].set_title('two variable plot')
ax[1].legend()
plt.show()
