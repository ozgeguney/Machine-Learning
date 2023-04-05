import numpy as np
A = 5
B = [2,4,6]
print(np.dot(A,B))

A = [1,2,3]
B = [2,4,6]
# 1*2 + 2*4 + 3*6
print(np.dot(A, B))

A = 5
B = [[6, 7],
     [8, 9]]
print(np.dot(A, B))

#number of colums in the first matrix must be equal to number of rows in the second matrix
A = [1,2]
B = [[6, 7],
     [8, 9]]
#1.row * 1.column ---- 1.row*2.column
print("np.dot(A, B)",np.dot(A, B))
print("np.dot(B, A)", (np.dot(B, A)))

A = [[1, 2],
     [3, 4]]
B = [[6, 7],
     [8, 9]]

print(np.dot(A,B))
print(np.dot(B,A))
print(np.matmul(A,B))
print(np.matmul(B,A))

#matmul ==> With this method, we can’t use scalar values for our input.
A = [1,2]
B = [[6, 7],
     [8, 9]]
# this equals np.dot(a,b)
#print(np.matmul(A, B))
#print(np.matmul(B,A))
A = [[1, 2],
     [3, 4]]
B = [[6, 7],
     [8, 9]]
#this equals np.dot(a,b)
#print(np.matmul(A, B))

