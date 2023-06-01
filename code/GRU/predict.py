import os
import sys
sys.path.append('/home/yuda/Motion-pattern-recognition/code')
# 建立训练集
import recognition.build_dataset as bd
# 数据处理
import numpy as np
import pandas as pd
# 数据平滑
from recognition.build_dataset import smooth_data
# 分类报告
from sklearn import metrics
# 分类器
from sklearn.svm import SVC
# pdr定位
import location.pdr as pdr
# 画图
import matplotlib.pyplot as plt
from matplotlib import rcParams
# 神经网络
from model import GRUNet
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable

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
        b = np.array(b)
        train_Y.append(b)
    train_X_np = np.array(train_X, dtype="float64")
    train_Y_np = np.array(train_Y, dtype="float64")
    train_Y_np = train_Y_np.squeeze()
    train_x = torch.as_tensor(torch.from_numpy(train_X_np), dtype=torch.float32)
    train_y = torch.as_tensor(torch.from_numpy(train_Y_np), dtype=torch.float32)
    #train_x.requires_grad = True
    #train_y.requires_grad = True
    return train_x, train_y

def save_load_remainFeature(function, path, remain_features=None):
    '''
    保存或加载筛选出的特征
    '''
    if function == "save":
        writer = pd.ExcelWriter(path)  
        pd.DataFrame(remain_features).to_excel(writer,'page_1',float_format='%.5f')  #float_format 控制精度，将data_df写到表格的第一页中。若多个文件，可以在page_2中写入
        writer.save()
        return 
    elif function == "load":
        if not os.path.isfile(path):
            print("文件不存在")
        else:
            remain_features = list(pd.read_excel(path).iloc[:, 1])
            return remain_features
    else:
        print("请输入save或load")
        return


def ave_accuracy(predictions, labels):
    '''
    计算欧氏距离平均误差
    '''
    if predictions.shape[0] > labels.shape[0]:
        predictions = predictions[:labels.shape[0], :]
    elif predictions.shape[0] < labels.shape[0]:
        labels = labels[:predictions.shape[0], :]
    error = np.sum((predictions - labels)**2, 1)**0.5*0.6
    accuracy = np.mean(error)
    return round(accuracy, 3), error

# 训练数据准备
## 运动模式识别部分
### 数据准备
train_path = "./data/TrainData"
supple_train_path = "./data/TestData/" # 将部分连续运动状态下测得的数据用于训练

test_path = './data/TestData/exp1'
realTrace_path = './data/TestData/test_coordinate.csv'
# 保存或加载lgb筛选出的特征
save_load_path = './code/RemainFeature/lgb_select_feature.xlsx' 
# 训练好的CNN参数
CNNParameter_Path = './code/CNNParameter/net_1conv_10kernels_13KernelSize_100Batch.pkl'
freq = 25 # 数据采样频率是25Hz
label_coding = {'stand': 0, 'walk': 1, 'up': 2, 'down': 3}
feature_num = 44
training_dimention = feature_num + 1
startidx = 75 # 舍掉前75个点
window_wide = int(1.5 * freq) # 滑动窗口宽度

### 读入数据
train_set, all_feature_name = bd.creat_training_set(train_path, label_coding, startidx, window_wide, training_dimention)

test_set = bd.creat_testing_set(test_path, label_coding, startidx, freq, window_wide, training_dimention)
train_x, train_y = train_set[:, 0:feature_num], train_set[:, -1]
test_x, true_y = test_set[:, 0:feature_num], test_set[:, -1]
realTrace_df = pd.read_csv(realTrace_path)
realTrace = realTrace_df.loc[:, 'x':'z'].values

## 特征选择
df_train_x = pd.DataFrame(data=train_x, columns=all_feature_name) # 将训练集转化为datafram格式,作为feature_selector输入

#### 加载筛选出的特征
remain_features = save_load_remainFeature("load", save_load_path)

train_x = df_train_x[remain_features].values # 筛选特征后的训练集
df_test_x = pd.DataFrame(data=test_x, columns=all_feature_name) # 将测试集转化为datafram格式
test_x = df_test_x[remain_features].values # 筛选特征后的测试集

## SVM分类
svc = SVC(C=300, kernel="rbf", max_iter=1000)
svc.fit(train_x,train_y)
 
predictions_svc = svc.predict(test_x)
print('SVM分类报告: \n', metrics.classification_report(true_y, predictions_svc ))

# pdr定位
## 数据准备
walking_data_file = test_path + '/pdr_data.csv'
df_walking = pd.read_csv(walking_data_file)

linear = df_walking[[col for col in df_walking.columns if 'linear' in col]].dropna().values[startidx:]
gravity = df_walking[[col for col in df_walking.columns if 'gravity' in col]].dropna().values[startidx:]
rotation = df_walking[[col for col in df_walking.columns if 'rotation' in col]].dropna().values[startidx:]
gyro = df_walking[[col for col in df_walking.columns if 'gyr' in col]].dropna().values[startidx:]

## 数据平滑
linear = smooth_data(linear)
gravity = smooth_data(gravity)
rotation = smooth_data(rotation)
gyro = smooth_data(gyro)

pdr = pdr.Model(linear, gravity, rotation, gyro, CNNParameterPath=CNNParameter_Path)

## 通过添加高斯噪声获取训练数据
### 确定添加噪声的标准差
length_sigma_list = [0.01, 0.02, 0.03, 0.04, 0.05] # 单位m
angle_sigma_list = [1*np.pi/180, 2*np.pi/180, 3*np.pi/180,4*np.pi/180, 5*np.pi/180] # 单位弧度
h_sigma_list = [0.01, 0.02, 0.03, 0.04, 0.05] # 单位m
train_motion_vector = []
train_realTrace = np.zeros((0,3))
for i in range(len(length_sigma_list)):
    X_pdr_noise, Y_pdr_noise, Z_pdr_noise, strides_noise, angle_noise, delt_h_noise = \
                            pdr.pdr_position(frequency=freq, walkType='normal', \
                            offset = 0,initPosition=(0, 0, 0),fuse_oritation = False, \
                            predictPattern=predictions_svc, m_WindowWide=window_wide, \
                            addNoise=True, length_sigma=length_sigma_list[i], \
                            angle_sigma=angle_sigma_list[i], h_sigma=h_sigma_list[i],\
                            random_seed = 123)
    x = np.array(X_pdr_noise).reshape(-1, 1)
    y = np.array(Y_pdr_noise).reshape(-1, 1)
    z = np.array(Z_pdr_noise).reshape(-1, 1)
    pdr_predict = np.concatenate((x, y, z), axis=1)
    mean_pdr_error, pdr_error = ave_accuracy(pdr_predict, realTrace)
    print(f"第{i}个噪声PDR平均定位误差:{mean_pdr_error} m")

    for k, v in enumerate(angle_noise[:realTrace.shape[0]]):
        train_x = X_pdr_noise[k]
        train_y = Y_pdr_noise[k]
        train_z = Z_pdr_noise[k]
        train_l = strides_noise[k]
        train_theta = angle_noise[k]
        train_sin_theta = np.sin(train_theta)
        train_cos_theta = np.cos(train_theta)
        train_h = delt_h_noise[k]
        train_motion_vector.append([train_x,train_y,train_z,train_l,train_sin_theta,train_cos_theta,train_h])
    
    train_realTrace = np.concatenate((train_realTrace, realTrace), axis=0)

### 准备GRU训练数据输入
time_step = 2
train_motion_vector = np.array(train_motion_vector)
train_data, train_label = create_dataset(train_motion_vector, train_realTrace, time_step, 1)
train_dataset = TensorDataset(train_data, train_label)
train_loader = DataLoader(dataset=train_dataset,
                         batch_size=50,
                         shuffle=True)

### 初始化GRU
LR = 0.01
EPOCH = 100
gru = GRUNet(input_size = 7 , hidden_size = 64, output_size = 3, num_layers = 1, dropout = 0)
print(gru)

### 训练
optimizer = torch.optim.Adam(gru.parameters(), lr=LR)   
loss_func = nn.MSELoss()
hidden = None
torch.autograd.set_detect_anomaly(True)
### gpu加速
if torch.cuda.is_available():
    gru = gru.cuda()
    loss_func = loss_func.cuda()

for epoch in range(EPOCH):
    train_loss = 0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        inputs, labels = Variable(inputs), Variable(labels)
        prediction = gru(inputs, hidden)               
        loss = loss_func(prediction, labels)   
        optimizer.zero_grad()
        loss.backward(retain_graph=True)               
        optimizer.step()                
        train_loss += loss
        if (epoch + 1) % 10 == 0 and (i+1)%5 == 0:
            loss_onCPU = loss.cpu()
            print('第{}轮, 训练Loss:{:.2f}'.format(epoch+1, loss_onCPU.data.numpy() / i))
            

### 测试
#### 准备测试数据
X_pdr, Y_pdr, Z_pdr, strides, angle, delt_h = \
                        pdr.pdr_position(frequency=freq, walkType='normal', \
                        offset = 0,initPosition=(0, 0, 0),fuse_oritation = False, \
                        predictPattern=predictions_svc, m_WindowWide=window_wide)

x = np.array(X_pdr).reshape(-1, 1)
y = np.array(Y_pdr).reshape(-1, 1)
z = np.array(Z_pdr).reshape(-1, 1)
pdr_predict_test = np.concatenate((x, y, z), axis=1)
mean_pdr_error_test, pdr_error_test = ave_accuracy(pdr_predict_test, realTrace)
print(f"测试集修正前PDR平均定位误差:{mean_pdr_error_test} m")
test_motion_vector = []
for k, v in enumerate(angle[:realTrace.shape[0]]):
    test_x = X_pdr[k]
    test_y = Y_pdr[k]
    test_z = Z_pdr[k]
    test_l = strides[k]
    test_theta = angle[k]
    test_sin_theta = np.sin(train_theta)
    test_cos_theta = np.cos(train_theta)
    test_h = delt_h_noise[k]
    test_motion_vector.append([test_x,test_y,test_z,test_l,test_sin_theta,test_cos_theta,test_h])

### 准备GRU测试数据输入
test_motion_vector = np.array(test_motion_vector)
test_data, test_label = create_dataset(test_motion_vector, realTrace, time_step, 1)
if torch.cuda.is_available():
    test_data = test_data.cuda()
gru_prediction = gru(test_data,hidden)
gru_prediction = gru_prediction.cpu().detach().numpy()
gru_correct_predict = np.concatenate((pdr_predict_test[0:time_step],gru_prediction), axis=0)
mean_gru_error_test, gru_error_test = ave_accuracy(gru_correct_predict, realTrace)
print(f"测试集修正后PDR平均定位误差:{mean_gru_error_test} m")
