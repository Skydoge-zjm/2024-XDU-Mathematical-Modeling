import matplotlib.pyplot as plt

from packer import *
from LSTMModel import LSTM


sequence_length = 64
feature_list = ['LC', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos']
device = "cuda" if torch.cuda.is_available() else "cpu"

model = LSTM(input_size=7, hidden_size=16, output_size=1, num_layers=1)
state_dict = torch.load('./model/model_state_dict_2.pth', map_location=device)
model.load_state_dict(state_dict)
model.to(device)

start = 277
end = 9490

seq, scaler = packer_1(pred_ind=start,
                       sequence_length=sequence_length,
                       feature_name_list=feature_list,
                       need_scaler=1, whole_length=end-start)

start = 0
end = end - start
# print(seq)
print(seq.shape)
indices = [i for i in range(start, end)]
values = []

# 使用zip函数将索引和值配对，然后遍历更新
flag = 0
for i in range(start, end):
    x = seq[:, i: i + sequence_length, :]
    x0 = x.reshape((1, sequence_length, len(feature_list)))
    y_pred = model(x0.to(device)).cpu().detach().numpy()

    y_pred_value = y_pred[0, 0]
    y_pred_tensor = torch.tensor(y_pred_value, dtype=seq.dtype)
    try:
        seq[0, i+sequence_length, 0] = y_pred_tensor
    except:
        break
    y_pred = scaler.inverse_transform(y_pred)
    values.append(y_pred[0, 0])
    if (i - start + 1) % 100 == 0:
        print(f'{i - start + 1} / {end - start} finished.')

print(values)
print("Prediction finished, plotting...")

copy_file = './data/q1/lstm_predict_copy.xlsx'
outcome = './data/q1/lstm_predict_outcome.xlsx'

df = pd.read_excel(copy_file)

for idx, val in zip(indices, values):
    df.loc[df.index[idx], 'LC'] = val

df.to_excel(outcome, index=False)

df = pd.read_excel(outcome, index_col='DateTime')

plt.figure(figsize=(30, 5))
plt.plot(df.index, df['LC'])
plt.title('Line Current Over Time')
plt.xlabel('Date/Time')
plt.ylabel('Line Current')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('Line Current Over Time_2.png')
plt.show()

