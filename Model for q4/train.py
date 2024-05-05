import csv

import torch
from torch import nn
from torch.optim import SGD, AdamW, RMSprop, Adam
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from packer import *
from LSTMModel import LSTM


batch_size = 128
sequence_length = 512
epochs = 200

device = "cuda" if torch.cuda.is_available() else "cpu"
model = LSTM(input_size=7, hidden_size=16, output_size=1, num_layers=1).to(device)
optimizer = AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
loss_function = nn.MSELoss()

feature_list = ['AT2', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos']

axis_x = range(epochs)
axis_y1 = []
axis_y2 = []
pred_list = []
true_list = []

best_model = None
best_loss = 1000.0
fin_loss = 0


print(f'train on {device}')
for epoch in range(epochs):
    seq, labels = packer(batch_size=batch_size,
                         sequence_length=sequence_length,
                         feature_name_list=feature_list,
                         a=0.7, flag=0)

    optimizer.zero_grad()
    y_pred = model(seq.to(device))
    single_loss = loss_function(y_pred, labels.to(device))
    axis_y1.append(single_loss.item())
    single_loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0 or (epoch+1) <= 10:
        print(f'epoch: {epoch+1} Train loss: {single_loss.item()}')

    model.eval()
    with torch.no_grad():
        seq, labels = packer(batch_size=batch_size,
                             sequence_length=sequence_length,
                             feature_name_list=feature_list,
                             a=0.7, flag=1)
        y_pred = model(seq.to(device))
        single_loss = loss_function(y_pred, labels.to(device))
        axis_y2.append(single_loss.item())

        if single_loss < best_loss:
            best_model = model.state_dict().copy()
            best_loss = single_loss.item()

        if epoch == epochs - 1:
            model.load_state_dict(best_model)

            # 用最佳模型参数，计算最终损失
            seq, labels = packer(batch_size=batch_size,
                                 sequence_length=sequence_length,
                                 feature_name_list=feature_list,
                                 a=0.7, flag=1)
            y_pred = model(seq.to(device))
            single_loss = loss_function(y_pred, labels.to(device))
            fin_loss = loss_function(y_pred, labels.to(device)).item()

            # 测试集2020-7-1上画出真实-预测图
            seq, labels = packer(batch_size=batch_size,
                                 sequence_length=sequence_length,
                                 feature_name_list=feature_list,
                                 a=0.7, flag=1)
            y_pred = model(seq.to(device))
            label_list = labels.tolist()
            pred_list = y_pred.tolist()
            plt.figure(figsize=(10, 5))
            plt.title('Part of True and Predicted AT for 2020-7')
            plt.xlabel("Timestamp")
            plt.ylabel("Normalized Value")
            plt.plot(range(len(label_list)), label_list, color='red', label='True')
            plt.plot(range(len(pred_list)), pred_list, color='blue', label='Predicted')
            plt.legend(loc='upper right')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.savefig('True-Predict AT.png', format='png', dpi=500)
            print("saved figure True-Predict AT.png")

            torch.save(model.state_dict(), './model/model_state_dict_AT.pth')

        model.train()


filename = 'Loss AT.csv'

with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Train Loss', 'Test Loss'])
    writer.writerows(zip(axis_y1, axis_y2))


axis_y1 = gaussian_filter(np.array(axis_y1), sigma=1)
axis_y2 = gaussian_filter(np.array(axis_y2), sigma=1)
plt.figure()
plt.title('Training and Testing Loss  (fin_loss=' + str(fin_loss))
plt.xlabel("Iterations")
plt.ylabel("MSELoss")
plt.plot(axis_x, axis_y1, label='Train Loss', color='blue')
plt.plot(axis_x, axis_y2, label='Test Loss', color='orange')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('loss_plot AT.png', format='png', dpi=500)
