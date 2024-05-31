import numpy as np

# 定义矩阵 A
A = np.array([[4, -2], [1, 1]])

# 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(A)

print("eigenvalues:", eigenvalues)
print("eigenvectors:", eigenvectors)
