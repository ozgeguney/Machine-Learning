import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1.0, 2.0, 3.0,4.0, 5.0])
y_train = np.array([300.0, 500.0, 600.0, 700.0, 800.0])
m = x_train.shape[0]
#m2 = len(x_train)

# print(f"x train : {x_train}")
# print(f"y train : {y_train}")
# print(f"number of input is {m}")
# print(y_train[1])


w=160
b=100
# f_wb_0 = w*x_train[0] + b
# f_wb_1 = w*x_train[1] + b
#
# print(f_wb_0,f_wb_1)

def compute_model_output(x,w,b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w*x[i] + b
    return f_wb

prediction = compute_model_output(x_train,w,b)
plt.plot(x_train, prediction, c="b", label = "Our Prediction")
plt.scatter(x_train, y_train, c="r", marker="x", label = "Actual Values")
plt.title("House Pricing")
plt.xlabel("Size")
plt.ylabel("Price")
plt.legend()
plt.show()

x_new = 3.5
f_wb_new = w* x_new + b
print(f"${f_wb_new:0f} thousands dolar")


