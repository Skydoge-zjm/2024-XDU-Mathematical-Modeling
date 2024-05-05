import os

import numpy as np
import pandas as pd


def train_set_generate():
    month_train = ['7', '8', '9']
    dfs = []
    for month in month_train:
        root_path = f'./data/q4/data-2019-{month}.xlsx'
        df = pd.read_excel(root_path)
        df.rename(columns={'Date/Time': 'DateTime', 'Ambient Temperature OTLM [掳C] DEV2': 'AT2'}, inplace=True)
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df = df.set_index('DateTime')
        dfs.append(df)

    merged_df = pd.concat(dfs)
    merged_df.sort_index(inplace=True)
    df_resampled = merged_df.resample('5min').mean()
    df_interpolated = df_resampled.interpolate(method='linear')
    filename = 'lstm_train.xlsx'
    path = os.path.join('.', 'data', 'q4', filename)
    df_interpolated.to_excel(path)


def test_set_generate():
    month_train = ['6', '7']
    dfs = []
    for month in month_train:
        root_path = f'./data/q4/data-2020-{month}.xlsx'
        df = pd.read_excel(root_path)
        df.rename(columns={'Date/Time': 'DateTime', 'Ambient Temperature OTLM [掳C] DEV2': 'AT2'}, inplace=True)
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df = df.set_index('DateTime')
        dfs.append(df)

    merged_df = pd.concat(dfs)
    merged_df.sort_index(inplace=True)
    df_resampled = merged_df.resample('5min').mean()
    df_interpolated = df_resampled.interpolate(method='linear')
    filename = 'lstm_test.xlsx'
    path = os.path.join('.', 'data', 'q4', filename)
    df_interpolated.to_excel(path)


def time_encoding(i='train'):
    df = pd.read_excel(f'./data/q4/lstm_{i}.xlsx')
    df['hour_sin'] = np.sin(df['DateTime'].dt.hour * (2. * np.pi / 24))
    df['hour_cos'] = np.cos(df['DateTime'].dt.hour * (2. * np.pi / 24))
    df['day_sin'] = np.sin(df['DateTime'].dt.dayofweek * (2. * np.pi / 7))
    df['day_cos'] = np.cos(df['DateTime'].dt.dayofweek * (2. * np.pi / 7))
    df['month_sin'] = np.sin((df['DateTime'].dt.month - 1) * (2. * np.pi / 12))
    df['month_cos'] = np.cos((df['DateTime'].dt.month - 1) * (2. * np.pi / 12))
    df.to_excel(f'./data/q4/lstm_{i}.xlsx')











if __name__ == '__main__':
    # train_set_generate()
    # test_set_generate()
    time_encoding(i='test')


"""
train_data = data['2019-07-01':'2019-09-30']
test_data = data['2020-06-01':'2020-07-31']
# data = pd.read_excel(file_path, index_col='DateTime')
# data.index = pd.to_datetime(data.index)
# 归一化
scaler = MinMaxScaler(feature_range=(-1, 1))
train_normalized = scaler.fit_transform(train_data.values.reshape(-1, 1))
test_normalized = scaler.transform(test_data.values.reshape(-1, 1))

# 转换为监督学习问题
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq

# 定义时间窗口
time_window = 5
train_inout_seq = create_inout_sequences(train_normalized, time_window)
test_inout_seq = create_inout_sequences(test_normalized, time_window)

# 创建数据集
class TimeSeriesDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence, label = self.sequences[idx]
        return torch.FloatTensor(sequence), torch.FloatTensor(label)

# 创建数据加载器
train_dataset = TimeSeriesDataset(train_inout_seq)
test_dataset = TimeSeriesDataset(test_inout_seq)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
"""