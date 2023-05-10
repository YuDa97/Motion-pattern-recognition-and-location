import numpy as np
from preparedata import LoadData
from CNN import CNN 
from sklearn.model_selection import train_test_split
import torch 
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
import pandas as pd
from tensorboardX import SummaryWriter    
import matplotlib.ticker as mtick
from matplotlib import rcParams
from zlib import Z_HUFFMAN_ONLY
import matplotlib.pyplot as plt
import seaborn as sns
###############################################################################
def split_data(dataset, label):
    '''
    dataset: 3-D(step*feature*samples), ndarry
    label: 1-D, ndarry
    将一步的和两步的数据分开
    '''
    s = dataset.shape[0] # 步数
    row = dataset.shape[1]
    col = dataset.shape[2]
    one_step_dataset = np.zeros((0, row, col))
    one_step_label = np.zeros((0, 1))
    two_step_dataset = np.zeros((0, row, col))
    two_step_label = np.zeros((0, 1))
    for i in range(0, s):
        if label[i] == 0.34:
            one_step_dataset = np.concatenate((one_step_dataset, np.array([dataset[i]])))
            one_step_label = np.concatenate((one_step_label, np.array([[label[i]]])))
        else:
            two_step_dataset = np.concatenate((two_step_dataset, np.array([dataset[i]])))
            two_step_label = np.concatenate((two_step_label, np.array([[label[i]]])))

    return one_step_dataset, one_step_label, two_step_dataset, two_step_label

def cal_mse(pred_y, true_y):
    return np.mean(np.abs(pred_y - true_y))

def cal_std(pred_y, true_y):
    return np.std(np.abs(pred_y - true_y), ddof=1)




# 数据准备
dataPath_down = './data/SLEdata/down'
dataPath_up = './data/SLEdata/up'
CNNParameterPath = './code/CNNParameter/net_1conv_10kernels_13KernelSize_100Batch.pkl'
freq = 25
## 下楼
LD_down = LoadData(dataPath_down, freq)
dataset_down, label_down = LD_down.build_dataset()
print("下楼步数: " , dataset_down.shape[0])

## 上楼
LD_up = LoadData(dataPath_up, freq)
dataset_up, label_up = LD_up.build_dataset()
print("上楼步数: " , dataset_up.shape[0])
dataset_all, label_all = np.concatenate((dataset_down,  dataset_up), axis=0), \
                        np.concatenate((label_down, label_up), axis=0)
print("总步数:", dataset_all.shape[0])

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

###############################################################################
# 训练CNN步长预测模型
## 超参数
EPOCH = 100           # 训练整批数据多少次
BATCH_SIZE = 100
LR = 0.001 

X_train= torch.as_tensor(torch.from_numpy(X_train), dtype=torch.float32)
y_train = torch.as_tensor(torch.from_numpy(y_train), dtype=torch.float32)
X_test = torch.as_tensor(torch.from_numpy(X_test), dtype=torch.float32)
y_test = torch.as_tensor(torch.from_numpy(y_test), dtype=torch.float32)
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(dataset=train_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=True)

 
cnn = CNN()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   
#loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted
loss_func = nn.MSELoss()

# gpu加速
if torch.cuda.is_available():
    cnn = cnn.cuda()
    loss_func = loss_func.cuda()
# 训练和测试
## 训练
#loos_writer = SummaryWriter('runs/example')
'''
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):# 分配 batch data, normalize x when iterate train_loader
        if torch.cuda.is_available():
            b_x = b_x.cuda()
            b_y = b_y.cuda()
        output = cnn(b_x)            # cnn output
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients
        #loos_writer.add_scalar("loss", loss.detach(), epoch)
        if step % 50 == 0:
            loss_onCPU = loss.cpu()
            print('Epoch: ', epoch, '| train loss: %.4f' % loss_onCPU.data.numpy())
#loos_writer.close()
'''
## 预测
###加载训练参数

if torch.cuda.is_available():
    cnn.load_state_dict(torch.load(CNNParameterPath))
else:
    cnn.load_state_dict(torch.load(CNNParameterPath, map_location='cpu'))

if torch.cuda.is_available():
    X_test = X_test.cuda()
pred_y = cnn(X_test)
pred_y = pred_y.cpu().detach().numpy()
y_test = y_test.detach().numpy()

##计算指标
Error = np.abs(pred_y - y_test)
SLE_mse = cal_mse(pred_y, y_test)
SLE_std = cal_std(pred_y, y_test)
Max_error = np.max(Error)
Min_error = np.min(Error)
print(f"平均步长误差为:\n{SLE_mse} m")
print(f"标准差:\n{SLE_std} m")
print(f"最大误差为:\n{Max_error} m")
print(f"最小误差为:\n{Min_error} m")

# #将数据写入excel
# error_df = pd.DataFrame(Error)
# col_name = 'CNN'
# error_df.columns = [col_name]
# with pd.ExcelWriter('./runs/CNNParameterTuning/ResultCompare.xlsx', mode='a', if_sheet_exists='replace') as writer:
#     error_df.to_excel(writer, sheet_name=col_name, float_format='%.5f', columns=[col_name])

#保留网络参数
# torch.save(cnn.state_dict(),f'./code/CNNParameter/net_1conv_10kernels_13KernelSize_100Batch_Adam_0.001lr.pkl')

# 画图
def plot_mse_cdf(Error):
    config = {
    "font.family":'Times New Roman',  # 设置字体类型
    #     "mathtext.fontset":'stix',
    }
    rcParams.update(config)
    plt.figure(figsize=(20,10), dpi=100)
    plt.xticks(fontsize=25) #设置坐标轴刻度大小
    plt.yticks(fontsize=25)

    plt.grid(linestyle='-.') 
    ax=plt.gca()

    # 设置坐标轴的范围
    plt.xlim(0, 0.4)
    #plt.ylim(0, 1)
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))#设置横坐标刻度保留2位小数
    ax.spines['bottom'].set_linewidth(1.2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(1.2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(1.2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(1.2)
    ax.set_xlabel('MSE(cm)', fontsize=25)#设置横纵坐标标签
    ax.set_ylabel('Cumulative Probability Distribution ', fontsize=25)

    sns.ecdfplot(Error,linewidth=2)

    plt.legend(fontsize = 20,bbox_to_anchor=(0.85,0.8)) #显示图例，字体为20
    plt.show()
    #plt.savefig('./Figure/KernelNumError_cdf.jpg',format='jpg',bbox_inches = 'tight',dpi=300)

#plot_mse_cdf(Error)