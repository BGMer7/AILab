import numpy as np

# 标量
s = 5
print(s)

# 向量
v = np.array([1, 2])
print(v)

# 矩阵
m = np.array([[1, 3], [2, 4]])
print(m)

# 张量
t = np.array([[[1, 2, 3], [2, 3, 4]], [[3, 4, 5], [4, 5, 6]], [[5, 6, 7], [6, 7, 8]]])
print(t)

# 一个矩阵A
A = np.array([[1, 2, 3], [1, 0, 2]])
print(A)

# 矩阵A的转置
A_t = A.transpose()
print(A_t)

# 矩阵A
A = np.array([[1, 2, 3], [4, 5, 6]])

# 矩阵A1
A1 = np.array([[1, 2, 3], [2, 3, 4]])

# 矩阵B
B = np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7]])

C = A.dot(B)  # 点乘 矩阵乘积
C = np.dot(A, B)
print(A.dot(B) == np.dot(A, B))  # 两种写法均可
# print(np.dot(A, B) == np.dot(B, A))  # 矩阵不满足交换律

print("matrix hadamard product \n", np.multiply(A, A1))
print("matrix hadamard product \n", A * A1)
# [[1  4  9]
#  [8 15 24]]

v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
v = v1.dot(v2)  # 向量内积  结果为标量
v = np.dot(v1, v2)
v = np.dot(v2, v1)  # 满足交换律
print("向量内积: ", v)
# 32=4+10+18
