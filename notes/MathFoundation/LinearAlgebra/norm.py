import numpy as np

# 定义一个向量
v = np.array([1, 0, 3, 0, 0, 2, 4])

# 计算0范数
l0_norm = np.count_nonzero(v)
print(f"0 norm: {l0_norm}")

# 计算L1范数
l1_norm = np.sum(np.abs(v))
print(f"L1 norm: {l1_norm}")

# 计算L2范数
l2_norm = np.linalg.norm(v)
print(f"L2 norm: {l2_norm}")

# 定义矩阵 A
A = np.array([[1, -2, 3], [-4, 5, -6], [7, -8, 9]])

# 计算每一行元素的绝对值之和
row_sums = np.sum(np.abs(A), axis=1)

# 计算无穷范数
infinity_norm = np.max(row_sums)
print("A infinity:", infinity_norm)
