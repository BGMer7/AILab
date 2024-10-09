## Attention Pooling

**Attention Pooling by Similarity** 是一种基于相似度的注意力汇聚方法，它将传统的注意力机制与特征汇聚（pooling）结合在一起，通过计算相似度来为输入序列中的每个元素分配不同的权重，从而实现更加灵活和自适应的汇聚方式。以下是对这个概念的理解：

Attention Pooling by Similarity 的核心在于根据输入序列中各个元素与一个或多个**查询向量（Query）**的相似度，为每个元素分配注意力权重。通过这些权重加权输入序列中的元素，可以实现对特征的动态汇聚。这种方式与传统的最大池化（Max Pooling）或平均池化（Average Pooling）不同，注意力池化是基于输入元素与查询之间的关系动态生成的，而不是固定的汇聚方式。