import os
import tarfile
from urllib import request
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32


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
    housing_data.hist(bins=50, figsize=(20,15))
    plt.savefig(housing_path)
    # plt.show()


def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


housing_data = load_housing_data()
# draw_and_save_housing_data(housing_data)
train_set, test_set = split_train_test(housing_data, 0.2)

