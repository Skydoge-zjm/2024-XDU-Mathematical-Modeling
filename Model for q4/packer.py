import random

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class DataPacker:
    def __init__(self, mydata, feature_list, a):
        self.data = mydata
        self.feature_list = feature_list
        self.a = a
        self.x = None
        self.y = None
        self.split_indices = None

    def pack(self, batch_size=256, sequence_length=32, flag=0):
        self.x = np.zeros((batch_size, sequence_length, len(self.feature_list)))
        self.y = np.zeros((batch_size, 1))
        self.random_choice(batch_size=batch_size, sequence_length=sequence_length, flag=flag)
        return self.x, self.y

    def pack_1(self, pred_ind, sequence_length=32, whole_length=None):
        start = pred_ind - sequence_length
        self.x = self.data[start: start + sequence_length + whole_length, :]
        self.x = self.x.reshape((1, sequence_length + whole_length, len(self.feature_list)))
        return self.x

    def random_choice(self, batch_size, sequence_length, flag=0):
        samples = random.choices(range(len(self.data)), k=batch_size)
        if flag == 0 or flag == 1:
            for i in range(batch_size):
                # 在data_i的长度内中选取连续的sequence_length行数据
                start = random.randint(0, self.data.shape[0] - sequence_length - 1)
                selected_x = self.data[start: start + sequence_length, :]
                selected_y = self.data[start + sequence_length, 0]
                selected_x = selected_x.reshape(1, sequence_length, len(self.feature_list))
                selected_y = np.array([[selected_y]])

                if i == 0:
                    self.x = selected_x
                    self.y = selected_y
                else:
                    self.x = np.concatenate((self.x, selected_x), axis=0)
                    self.y = np.concatenate((self.y, selected_y), axis=0)
        elif flag == 2:
            start = 8543  # 2020-7-1
            for i in range(batch_size):
                selected_x = self.data[start + i: start + i + sequence_length, :]
                selected_y = self.data[start + i + sequence_length, 0]
                selected_x = selected_x.reshape(1, sequence_length, len(self.feature_list))
                selected_y = np.array([[selected_y]])
                if i == 0:
                    self.x = selected_x
                    self.y = selected_y
                else:
                    self.x = np.concatenate((self.x, selected_x), axis=0)
                    self.y = np.concatenate((self.y, selected_y), axis=0)


def packer(batch_size=64, sequence_length=32, feature_name_list=None, a=0.7, flag=0, need_scaler=0):
    if feature_name_list is None:
        feature_name_list = []
    if flag == 0:
        df = pd.read_excel('./data/q4/lstm_train.xlsx', index_col='DateTime')[feature_name_list]
    else:
        df = pd.read_excel('./data/q4/lstm_test.xlsx', index_col='DateTime')[feature_name_list]

    scaler = MinMaxScaler()
    features = df[['AT2']]
    scaled_features = scaler.fit_transform(features)
    df[['AT2']] = scaled_features

    original_data = np.array(df)
    feature_list = [i for i in range(len(feature_name_list))]
    data1 = DataPacker(mydata=original_data, feature_list=feature_list, a=a)
    x, y = data1.pack(batch_size, sequence_length, flag=flag)
    x = torch.from_numpy(x.astype(float)).to(torch.float32)
    y = torch.from_numpy(y.astype(float)).to(torch.float32)
    if need_scaler == 0:
        return x, y
    else:
        return x, y, scaler


def packer_1(pred_ind, sequence_length=64, feature_name_list=None, a=0.7, need_scaler=0, whole_length=None):
    df = pd.read_excel('./data/q4/lstm_predict.xlsx', index_col='DateTime')[feature_name_list]
    feature_list = [i for i in range(len(feature_name_list))]

    scaler = MinMaxScaler()
    features = df[['AT2']]
    scaled_features = scaler.fit_transform(features)
    df[['AT2']] = scaled_features

    original_data = np.array(df)
    data1 = DataPacker(mydata=original_data, feature_list=feature_list, a=a)
    x = data1.pack_1(pred_ind, sequence_length, whole_length)
    x = torch.from_numpy(x.astype(float)).to(torch.float32)

    if need_scaler == 0:
        return x
    else:
        return x, scaler
