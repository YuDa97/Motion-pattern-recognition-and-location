import numpy as np
from preparedata import LoadData
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.ticker as mtick
from matplotlib import rcParams
from zlib import Z_HUFFMAN_ONLY
import matplotlib.pyplot as plt
import seaborn as sns

def Transform_data(data):
    TransData = np.array([])
    for v in data:
        TransData = np.append(TransData, np.array(v))
    return TransData
def split_data(dataset, label):
    '''
    dataset: 1-D, dic
    label: 1-D, ndarry
    将一步的和两步的数据分开
    '''
    s = dataset.shape[0] # 步数
    one_step_dataset = np.zeros((0))
    one_step_label = np.zeros((0, 1))
    two_step_dataset = np.zeros((0))
    two_step_label = np.zeros((0, 1))
    for i in range(0, s):
        if label[i] == 0.34:
            one_step_dataset = np.concatenate((one_step_dataset, np.array([dataset[i]])))
            one_step_label = np.concatenate((one_step_label, np.array([[label[i]]])))
        else:
            two_step_dataset = np.concatenate((two_step_dataset, np.array([dataset[i]])))
            two_step_label = np.concatenate((two_step_label, np.array([[label[i]]])))

    return one_step_dataset, one_step_label, two_step_dataset, two_step_label
def NSL(step_info):
    '''
    步长推算
    NSL:
    k为身高相关常数
    v为每步的数组
    '''
    predict = np.zeros((0, 1))
    for v in step_info:
        k = 0.28
        step_length =  np.array([np.power(v['acceleration'] - v['v_acceleration'], 1/4) * k])
        predict = np.concatenate((predict, np.array([step_length])))
    return predict

def cal_mse(pred_y, true_y):
    return np.mean(np.abs(pred_y - true_y))
# 数据准备
dataPath_down = './data/SLEdata/down'
dataPath_up = './data/SLEdata/up'
freq = 25

LD_down = LoadData(dataPath_down, freq)
LD_up = LoadData(dataPath_up, freq)
_, label_down = LD_down.build_dataset()
_, label_up = LD_up.build_dataset()


down_steps = LD_down.step_info[1:] # 为了与CNN数据集对齐，CNN数据需要步频信息，跳过了第一步
up_steps = LD_up.step_info[1:]

print(f"上楼步数: {len(up_steps)}")
print(f"下楼步数: {len(down_steps)}")

down_steps = Transform_data(down_steps)
up_steps = Transform_data(up_steps)

dataset_all, label_all = np.concatenate((down_steps,  up_steps), axis=0), \
                        np.concatenate((label_down, label_up), axis=0)


## 划分一步和两步的数据集
one_step_dataset, one_step_label, \
    two_step_dataset, two_step_label = split_data(dataset_all, label_all)

## 分开按比例抽样 组成训练集和测试集
X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(
    one_step_dataset, one_step_label, test_size=0.2, random_state=42)

X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
    two_step_dataset, two_step_label, test_size=0.2, random_state=42)

X_train = np.concatenate((X_train_o, X_train_t))
X_test = np.concatenate((X_test_o, X_test_t))
y_train= np.concatenate((y_train_o, y_train_t))
y_test = np.concatenate((y_test_o, y_test_t))

print("训练集数量:", X_train.shape[0])
print("测试集数量:", y_test.shape[0])

pred_y = NSL(X_test)
##计算指标
Error = np.abs(pred_y - y_test)
SLE_mse = cal_mse(pred_y, y_test)
Max_error = np.max(Error)
Min_error = np.min(Error)
print(f"平均步长误差为:\n{SLE_mse} m")
print(f"最大误差为:\n{Max_error} m")
print(f"最小误差为:\n{Min_error} m")

#将数据写入excel
error_df = pd.DataFrame(Error)
col_name = 'NSL'
error_df.columns = [col_name]
with pd.ExcelWriter('./runs/CNNParameterTuning/ResultCompare.xlsx', mode='a', if_sheet_exists='replace') as writer:
    error_df.to_excel(writer, sheet_name=col_name, float_format='%.5f', columns=[col_name])