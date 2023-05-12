from model import GRUNet
import numpy as np
import torch
from torch import nn

def create_dataset(data, label, time_step, n_next):
    '''
    对数据进行处理,创建用于训练和测试gru的数据集
    time_step:前n个数据
    n_next:需要预测的后n个数据
    '''

    train_X, train_Y = [], []
    for i in range(data.shape[0]-time_step-n_next+1):
        a = data[i:(i+time_step),:]
        train_X.append(a)
        tempb = label[(i+time_step):(i+time_step+n_next),:]
        b = []
        for j in range(len(tempb)):
            b.append(tempb[j])
        train_Y.append(b)
    train_X_np = np.array(train_X, dtype="float64")
    train_Y_np = np.array(train_Y, dtype="float64")
    train_Y_np = train_Y_np.squeeze()
    train_x = torch.as_tensor(torch.from_numpy(train_X_np), dtype=torch.float32)
    train_y = torch.as_tensor(torch.from_numpy(train_Y_np), dtype=torch.float32)
    #train_x.requires_grad = True
    #train_y.requires_grad = True
    return train_x, train_y