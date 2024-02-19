import os
import tarfile
from urllib import request
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


# 该资源需要翻墙下载
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    # request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    housing_data = pd.read_csv(csv_path)
    print(housing_data.info())
    # "ocean_proximity" can be replaced by other column
    print(housing_data["ocean_proximity"].value_counts())
    print(housing_data.head())
    return housing_data


def draw_and_save_housing_data(housing_data, housing_path=HOUSING_PATH):
    housing_data.hist(bins=50, figsize=(20, 15))
    # plt.savefig(housing_path)
    # plt.show()


def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


print("----------------------------------------------------------------")
print("---------------------fetch & load data--------------------------")
print("----------------------------------------------------------------")
housing_data = load_housing_data()
# draw_and_save_housing_data(housing_data)

train_set, test_set = split_train_test(housing_data, 0.2)


def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xFFFFFFFF < test_ratio * 2**32


def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


print("-------------------行索引作为id------------------")
# adds an 'index' cloumn
housing_data_with_id = housing_data.reset_index()
train_set, test_set = split_train_test_by_id(housing_data_with_id, 0.2, "index")
print(train_set.head())
print(test_set.head())


print("-------------------经纬度作为id------------------")
housing_data_with_id["id"] = housing_data["longitude"] * 1000 + housing_data["latitude"]
train_set, test_set = split_train_test_by_id(housing_data_with_id, 0.2, "id")
print(train_set.head())
print(test_set.head())


# 如果每次都产生一个新的随机的训练集和测试集，这不是我们期望的情况
# 解决方案一：保留数据集，后续的工作只是在已有的数据集的基础上进行
# 解决方案二：在调用np.random.permutation()之前设置一个随机数生成器的种子（例如，np.random.seed(42)），从而让它始终生成相同的随机索引
# 见书2.3
train_set, test_set = train_test_split(housing_data, test_size=0.2, random_state=42)


def split_housing_data(housing_data):
    housing_data["income_cat"] = pd.cut(
        housing_data["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )
    # 直方图
    housing_data["income_cat"].hist()
    # plt.show()


split_housing_data(housing_data)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing_data, housing_data["income_cat"]):
    strat_train_set = housing_data.loc[train_index]
    strat_test_set = housing_data.loc[test_index]

# 收入的各种类型的比例
print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))


print("----------------------------------------------------------------")
print("---------------------visualize data-----------------------------")
print("----------------------------------------------------------------")
housing_data = strat_train_set.copy()
housing_data.plot(kind="scatter", x="longitude", y="latitude")
housing_data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)


housing_data.plot(
    kind="scatter",
    x="longitude",
    y="latitude",
    alpha=0.4,
    s=housing_data["population"] / 100,
    label="population",
    figsize=(10, 7),
    c="median_house_value",
    cmap=plt.get_cmap("jet"),
    colorbar=True,
)

plt.legend()

# 数据之间的相关性
corr_matrix = housing_data.corr(numeric_only=True)
corr_matrix["median_house_value"].sort_values(ascending=False)

# scatter_matrix绘制每个数值属性相对于其他数值属性的相关性
attributes = [
    "median_house_value",
    "median_income",
    "total_rooms",
    "housing_median_age",
]
scatter_matrix(housing_data[attributes], figsize=(12, 8))
housing_data.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
# plt.show()


print("----------------------------------------------------------------")
print("---------------------prepare data-------------------------------")
print("----------------------------------------------------------------")


# 用属性的中位数替换属性的缺失值
imputer = SimpleImputer(strategy="median")
housing_num = housing_data.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)
print(imputer.statistics_)
print("---------------------缺失值已替换-------------------------------")

X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)


rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[
                X, rooms_per_household, population_per_household, bedrooms_per_room
            ]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing_data.values)

# 用流水线来规定一个机器学习的顺序
num_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("attribs_adder", CombinedAttributesAdder()),
        ("std_scaler", StandardScaler()),
    ]
)

housing_num_tr = num_pipeline.fit_transform(housing_num)
