**Pure Sampling**（纯采样）在统计学和机器学习中是指从总体中以均匀和无偏的方式随机选取样本的过程，通常不受任何约束或调整因素的影响。在这个过程中，所有的样本都有相同的机会被选中，没有任何人为的偏差。

具体来说，Pure Sampling 可以分为几种不同的类型，常见的包括：

1. **简单随机抽样（Simple Random Sampling）**：
   - 从总体中随机选取样本，保证每个样本被选中的概率是相同的。
   - 例如，从一个装有 100 个球的袋子中随机抽取 10 个球，每个球被选中的机会相同。

2. **无放回抽样（Sampling Without Replacement）**：
   - 每个选中的样本在被选中后就不再返回总体中，避免重复选择。
   - 在从一个集合中抽取样本时，一旦一个样本被选中，它就不再参与后续的抽样。

3. **有放回抽样（Sampling With Replacement）**：
   - 每个选中的样本会被放回到总体中，因此同一个样本可能会被多次选中。

**Pure Sampling 的特点：**
- **随机性**：样本的选择完全基于随机过程，避免人为干预。
- **均匀性**：每个样本被选中的概率是相同的，确保了样本的代表性。
- **无偏性**：不引入任何系统的偏差，能更好地推断总体的特征。

### **应用**
1. **统计推断**：用于从样本数据推断总体属性，确保推断结果的有效性和准确性。
2. **机器学习**：在训练集的构建中，纯采样方法有助于消除样本选择偏差。
3. **蒙特卡罗方法**：在许多随机过程的模拟中，Pure Sampling 被广泛应用于生成随机样本。
