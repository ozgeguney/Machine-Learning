import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g

z_temp = np.arange(-10,10)
y = sigmoid(z_temp)

# np.set_printoptions(precision=3)
# print("Input (z), Output sigmoid (z)")
# print(np.c_[z_temp, y])

fig, ax = plt.subplots(1,1,figsize=(6,3))
ax.plot(z_temp, y, c="b")
ax.set_title("Sigmoid function")
ax.set_xlabel("z")
ax.set_ylabel("sigmoid(z)")
#plt.show()







