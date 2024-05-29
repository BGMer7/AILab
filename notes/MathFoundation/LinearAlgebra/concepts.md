# Concepts

![标量、向量、矩阵、张量.png](https://cdn.nlark.com/yuque/0/2020/png/1594055/1592031215265-92286523-aff1-46db-8976-612f24c10013.png?x-oss-process=image%2Fwatermark%2Ctype_d3F5LW1pY3JvaGVp%2Csize_41%2Ctext_Y2hteDA5MjlAdmlwLnFxLmNvbQ%3D%3D%2Ccolor_FFFFFF%2Cshadow_50%2Ct_80%2Cg_se%2Cx_10%2Cy_10%2Fformat%2Cwebp)

![1-3Dsignal.png](https://cdn.nlark.com/yuque/0/2020/png/1594055/1592031297509-08ab9130-8587-4486-b89b-6d872a40b87d.png?x-oss-process=image%2Fwatermark%2Ctype_d3F5LW1pY3JvaGVp%2Csize_30%2Ctext_Y2hteDA5MjlAdmlwLnFxLmNvbQ%3D%3D%2Ccolor_FFFFFF%2Cshadow_50%2Ct_80%2Cg_se%2Cx_10%2Cy_10%2Fformat%2Cwebp)



## scalar

一个单独的数，一般用斜体表示标量，通常是小写字母。


## vector

一列数，并且有序排列。


## matrix
一个二维数组，每个元素被两个索引唯一确定。


## tensor
多于二维的数组。
用大写字母A表示张量，$A_i,_j,_k$表示其元素。



```python
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

```



## positive definite & positive semi-definite

正定和半正定这两个词的英文分别是positive definite和positive semi-definite，其中，definite是一个形容词，表示“明确的、确定的”等意思。

【广义定义】特征值全是正实数的实对称矩阵为正定矩阵。

【广义定义】给定一个![img](https://cdn.nlark.com/yuque/__latex/607acaa73c762411b20745149a11e90b.svg)的实对称矩阵(即矩阵内元素都是实数的对称矩阵)![img](https://cdn.nlark.com/yuque/__latex/7fc56270e7a70fa81a5935b72eacbe29.svg)，若对于任意长度为![img](https://cdn.nlark.com/yuque/__latex/7b8b965ad4bca0e41ab51de7b31363a1.svg)的非零向量![img](https://cdn.nlark.com/yuque/__latex/9dd4e461268c8034f5c8564e155c67a6.svg)，有![img](https://cdn.nlark.com/yuque/__latex/0e9274ebc77deaa6f531e3b869a44e1d.svg)恒成立，则矩阵![img](https://cdn.nlark.com/yuque/__latex/7fc56270e7a70fa81a5935b72eacbe29.svg)是一个正定矩阵。

【狭义定义】给定一个![img](https://cdn.nlark.com/yuque/__latex/607acaa73c762411b20745149a11e90b.svg)的实对称矩阵(即矩阵内元素都是实数的对称矩阵)![img](https://cdn.nlark.com/yuque/__latex/7fc56270e7a70fa81a5935b72eacbe29.svg)，若对于任意长度为![img](https://cdn.nlark.com/yuque/__latex/7b8b965ad4bca0e41ab51de7b31363a1.svg)的非零向量![img](https://cdn.nlark.com/yuque/__latex/9dd4e461268c8034f5c8564e155c67a6.svg)，有![img](https://cdn.nlark.com/yuque/__latex/7ad84c565a4baef237aad7b5d4624725.svg)恒成立，则矩阵![img](https://cdn.nlark.com/yuque/__latex/7fc56270e7a70fa81a5935b72eacbe29.svg)是一个半正定矩阵。注意，正定要求是大于0，半正定可以等于0。



【例1】单位矩阵![img](https://cdn.nlark.com/yuque/__latex/f39428a0a5831eabd45ba0d5cb1eff6e.svg)是否是正定矩阵？

解：设向量![img](https://cdn.nlark.com/yuque/__latex/1d83d3361ac175e5b5a117c6c52acdc4.svg)为非零向量，则![img](https://cdn.nlark.com/yuque/__latex/7416498adb4a7097a65b901e83ca6cab.svg)。由于![img](https://cdn.nlark.com/yuque/__latex/43cd1f01fc40ff198193084a874be8ab.svg)，故![img](https://cdn.nlark.com/yuque/__latex/a61e46fbf5a63a06a71a0b165c3dd45a.svg)恒成立，即单位矩阵![img](https://cdn.nlark.com/yuque/__latex/f39428a0a5831eabd45ba0d5cb1eff6e.svg)是正定矩阵。类推，对于任意单位矩阵![img](https://cdn.nlark.com/yuque/__latex/38c4e9be3752300d49c9aae54d6438ac.svg)，给定非零向量![img](https://cdn.nlark.com/yuque/__latex/9dd4e461268c8034f5c8564e155c67a6.svg)，恒有![img](https://cdn.nlark.com/yuque/__latex/83a7f168f6253af15f936c0e4a44a278.svg)。所以，单位矩阵是正定矩阵(positive definite)。



【例2】实对称矩阵![img](https://cdn.nlark.com/yuque/__latex/0f147356111292dec70ff1f28ea9eace.svg)是否是正定矩阵？

解：设向量![img](https://cdn.nlark.com/yuque/__latex/88a263042453edeedc238bfff22c0097.svg)为非零向量，则![img](https://cdn.nlark.com/yuque/__latex/a65dd592df3a9b375b347e22a65a13ee.svg)，因此，矩阵![img](https://cdn.nlark.com/yuque/__latex/7fc56270e7a70fa81a5935b72eacbe29.svg)是正定矩阵。



## identity matrix

单位矩阵：所有的对角线元素都是1，其他的所有元素都是0。



## diagonal matrix

对角矩阵：对角矩阵(diagonal matrix)是一个主对角线之外的元素皆为0的矩阵，常写为$diag(a_1, a_2, ..., a_n)$。对角矩阵可以认为是矩阵中最简单的一种，值得一提的是，对角线上的元素可以为 0 或其他值，对角线上元素相等的对角矩阵称为数量矩阵，对角线上元素全为1的对角矩阵称为单位矩阵。对角矩阵的运算包括和、差运算、数乘运算、同阶对角阵的乘积运算，且结果仍为对角阵。

## trace

矩阵的迹：矩阵主对角线元素之和。

![img](https://cdn.nlark.com/yuque/__latex/c12fd4bd73c3aff952b8c34ecf69c901.svg)



## norm TODO

#### Frobenius norm 



# Operations

## Transpose

![img](https://cdn.nlark.com/yuque/__latex/dd008043b95edf3d4be664764d54baee.svg)    ![img](https://cdn.nlark.com/yuque/__latex/4a5e26f3251cb2937a01ba8793474521.svg)

```python
# 一个矩阵A
A = np.array([[1, 2, 3],[1, 0, 2]])
print(A)

# 矩阵A的转置
A_t = A.transpose()
print(A_t)

[[1 2 3]
 [1 0 2]]

[[1 1]
 [2 0]
 [3 2]]
```



## dot product

内积（Inner Product）和点乘（Dot Product）在数学和线性代数中通常是指同一种运算，但它们在不同的上下文中可能有细微差别。

向量内积：两个向量的对应元素相乘并相加。

$a \cdot b = |a|\cdot|b|\cdot cos\theta$，$\theta$为两个向量的夹角。

从上述表达式中我们可以看出，向量内积的物理含义就是**向量a在b方向上的投影长度**，若b的模是1，则$a \cdot b = |a|\cdot cos\theta$，该结论同样可以适用于矩阵相乘，在后面的矩阵线性变换中我们会用到该结论。



![img](https://cdn.nlark.com/yuque/__latex/249d629066fa0c8d1ca34d2214aaba16.svg)

在机器学习和数值计算中，两个矩阵点乘有时也被称为**逐元素乘积**（Hadamard Product），它是指两个相同形状的矩阵或向量的对应元素相乘的操作。



**两个向量的点积是标量**，标量转置的结果是自身。

自此我们可以通过矩阵乘积的方式来表示线性方程组$Ax = b$。其中A是一个矩阵，m行n列，x是一个求解n行的未知向量，b是一个已知向量。



```python
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
# [[1  4  9]  =  [[1 2 3]  * [[1 2 3]
#  [8 15 24]]     [4 5 6]]    [2 3 4]]

v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
v = v1.dot(v2)  # 向量内积  结果为标量
v = np.dot(v1, v2)
v = np.dot(v2, v1)  # 满足交换律
print("向量内积: ", v)
# 32=4+10+18
```



## inverse

逆矩阵定义：在线性代数中，给定一个![img](https://cdn.nlark.com/yuque/__latex/7b8b965ad4bca0e41ab51de7b31363a1.svg)阶方阵![img](https://cdn.nlark.com/yuque/__latex/7fc56270e7a70fa81a5935b72eacbe29.svg) ，若存在一![img](https://cdn.nlark.com/yuque/__latex/7b8b965ad4bca0e41ab51de7b31363a1.svg)阶方阵![img](https://cdn.nlark.com/yuque/__latex/9d5ed678fe57bcca610140957afab571.svg) ，使得![img](https://cdn.nlark.com/yuque/__latex/5faadc0fd1fc8b0d097b32de0882f293.svg) ，其中 ![img](https://cdn.nlark.com/yuque/__latex/51e30ff0f3ad7f4a08fb2aea5cbc037b.svg) 为![img](https://cdn.nlark.com/yuque/__latex/7b8b965ad4bca0e41ab51de7b31363a1.svg)阶单位矩阵，则称![img](https://cdn.nlark.com/yuque/__latex/7fc56270e7a70fa81a5935b72eacbe29.svg)是可逆的，且![img](https://cdn.nlark.com/yuque/__latex/9d5ed678fe57bcca610140957afab571.svg)是![img](https://cdn.nlark.com/yuque/__latex/7fc56270e7a70fa81a5935b72eacbe29.svg)的逆矩阵，记作![img](https://cdn.nlark.com/yuque/__latex/1ff4e7c4ea49e4f89fcea2a90968d87f.svg) 。

![img](https://cdn.nlark.com/yuque/__latex/ad6900bdf3e469589d477e5257f90b19.svg)

![img](https://cdn.nlark.com/yuque/__latex/aecffe82d3789ecb1e02fadca31f4af4.svg)



只有正方形![img](https://cdn.nlark.com/yuque/__latex/607acaa73c762411b20745149a11e90b.svg)的矩阵，亦即方阵，才可能、但非必然有逆矩阵。若方阵![img](https://cdn.nlark.com/yuque/__latex/7fc56270e7a70fa81a5935b72eacbe29.svg)的逆矩阵存在，则称![img](https://cdn.nlark.com/yuque/__latex/7fc56270e7a70fa81a5935b72eacbe29.svg)为非奇异方阵或可逆方阵。



## SVD

