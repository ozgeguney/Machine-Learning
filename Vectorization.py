import numpy as np

a = np.arange(10);
print(a)
print(f"a[-1] = {a[-1]}")
print("a[2:7:1] = ", a[2:7:1])

a = np.array([1,2,3,4,5,6])
print(f"a : {a}")
# negate elements of a
b = -a
print(f"b = -a : {b}")

# sum all elements of a, returns a scalar
b = np.sum(a)
print(f"b = np.sum(a) : {b}")

b = np.mean(a)
print(f"b = np.mean(a): {b}")

b = a**2
print(f"b = a**2      : {b}")

a = np.array([ 1, 2, 3, 4])
b = np.array([-1,-2, 3, 4])
print(f"Binary operators work element wise: {a + b}")

a = np.array([1, 2, 3, 4])
# multiply a by a scalar
b = 5 * a
print(f"b = 5 * a : {b}")

def my_dot(a,b):
    result =0
    for i in range(a.shape[0]):
        result += a[i]*b[i]
    return result
print(a,b)
import time
tic = time.time()
print(my_dot(a,b))
toc = time.time()
print(f"loop version duration: {1000*(toc-tic):.4f} ms ")
tic = time.time()
print(np.dot(a,b))
toc = time.time()
print(f"Vectorized version duration: {1000*(toc-tic):.4f} ms ")

# it is a 2 Dimensional array or matrix
X = np.array([[1],[2],[3],[4]])
print(X)
w = np.array([2])
print(w)
c = np.dot(X[3], w)
print(c)

a = np.zeros((1, 5))
print(f"a shape = {a.shape}, a = {a}")

a = np.zeros((2, 1))
print(f"a shape = {a.shape}, a = {a}")

a = np.random.random_sample((1, 1))
print(f"a shape = {a.shape}, a = {a}")

a = np.array([[5], [4], [3]]);
print(f" a shape = {a.shape}, np.array: a = {a}")

a = np.arange(6).reshape(-1, 2)
print(a)
print(a[2, 0]) #Access an element
print(a[2]) #access a row

a = np.arange(20).reshape(-1, 10)
print(a) # all values
print(a[0, 2:7:1]) # selected row, selected columns
print(a[:, 2:7:1]) # all rows, selected columns
print(a[:,:]) #all values
print(a[1,:]) #selected row
print(a[1]) #selected row
print(a[0:1])



