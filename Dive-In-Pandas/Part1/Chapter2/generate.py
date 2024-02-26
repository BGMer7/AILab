import pandas as pd
import numpy as np

df1 = pd.DataFrame(
    {
        "国家": ["中国", "美国", "日本"],
        "地区": ["亚洲", "北美", "亚洲"],
        "人口": [13.97, 3.28, 1.26],
        "GDP": [14.34, 21.43, 5.08],
    }
)
print("df", df1)
print("index", df1.index)
print("info", df1.info)
print(df1["人口"])

df2 = pd.DataFrame(
    {
        "A": 1.0,
        "B": pd.Timestamp("20130102"),
        "C": pd.Series(1, index=list(range(4)), dtype="float32"),
        "D": np.array([3] * 4, dtype="int32"),
        "E": pd.Categorical(["test", "train", "test", "train"]),
        "F": "foo",
    }
)
print(df2)

df3 = pd.DataFrame(
    {
        "A": 1.0,  # 全都是1.0
        "B": [1, 2, 3, 4],  # 长度必须和行数一样
        "C": pd.Series(
            1, index=list(range(4)), dtype="float32"
        ),  # 生成一个长度为4的list
        "D": np.array([3] * 4, dtype="int32"),
        "E": pd.Categorical(["test", "train", "test", "train"]),
        "F": "foo",
    }
)
print(df3)

s = pd.Series([14.34, 21.43, 5.08], name="gdp")

print(type(s))  # pandas.core.series.Series
print(type(df1))  # pandas.core.frame.DataFrame


pd.Series(["a", "b", "c", "d", "e"])
pd.Series(("a", "b", "c", "d", "e"))

# 由索引分别为a、b、c、d、e的5个随机浮点数数组组成
s = pd.Series(np.random.randn(5), index=["a", "b", "c", "d", "e"])
print(s)
print(s.index)  # 查看索引

s = pd.Series(np.random.randn(5))  # 未指定索引，则默认从0开始递增
print(s)

d = {'b': 1, 'a': 0, 'c': 2}
s = pd.Series(d)
print(s)

# 如果指定索引，则会按索引顺序，如有无法与索引对应的值，会产生缺失值
s = pd.Series(d, index=['b', 'c', 'd', 'a'])
print(s)

