import os
# 建立训练集
import recognition.build_dataset as bd
# 数据处理
import numpy as np
import pandas as pd
# 数据平滑
from recognition.build_dataset import smooth_data
# 分类报告
from sklearn import metrics
# 特征选择
from recognition.feature_selector import FeatureSelector
# 分类器
from sklearn.svm import SVC
# WiFi定位
import location.wifi as wifi
# pdr定位
import location.pdr as pdr
# EKF融合定位
import location.fusion as fusion
# 画图
import matplotlib.pyplot as plt
from matplotlib import rcParams

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
    accuracy = np.mean((np.sum((predictions - labels)**2, 1))**0.5)*0.6
    return round(accuracy, 3)

# 运动模式识别部分
## 数据准备
train_path = "./data/TrainData"
supple_train_path = "./data/FuseLocationTestData/" # 将部分连续运动状态下测得的数据用于训练
supple_train_sets = ["exp1", "exp1.5", "exp2", "exp2.5", "exp3", "exp4", "exp6"] # 使用哪些测试数据补充进训练集
test_path = './data/FuseLocationTestData/exp5'
realTrace_path = './data/FuseLocationTestData/test_coordinate.csv'
# 保存或加载lgb筛选出的特征
save_load_path = './code/RemainFeature/lgb_select_feature_for_FuseLocationExp5.xlsx' 
# 训练好的CNN参数
CNNParameter_Path = './code/CNNParameter/net_2conv_3KernelSize_100Batch.pkl' 
freq = 25 # 数据采样频率是25Hz
label_coding = {'stand': 0, 'walk': 1, 'up': 2, 'down': 3}
feature_num = 44
training_dimention = feature_num + 1
startidx = 75 # 舍掉前75个点
window_wide = int(1.5 * freq) # 滑动窗口宽度

### 读入数据
train_set, all_feature_name = bd.creat_training_set(train_path, label_coding, startidx, window_wide, training_dimention)
# 补充训练集数据
for dataset in supple_train_sets:
    path = supple_train_path + dataset
    supple_train = bd.creat_testing_set(path, label_coding, startidx, freq, window_wide, training_dimention)
    train_set = np.concatenate((train_set, supple_train), axis=0)

test_set = bd.creat_testing_set(test_path, label_coding, startidx, freq, window_wide, training_dimention)
train_x, train_y = train_set[:, 0:feature_num], train_set[:, -1]
test_x, true_y = test_set[:, 0:feature_num], test_set[:, -1]
realTrace_df = pd.read_csv(realTrace_path)
realTrace = realTrace_df.loc[:, 'x':'z'].values

## 特征选择
df_train_x = pd.DataFrame(data=train_x, columns=all_feature_name) # 将训练集转化为datafram格式,作为feature_selector输入
'''
fs = FeatureSelector(data = df_train_x, labels = train_y) 
fs.identify_collinear(correlation_threshold=0.8, one_hot=False)
fs.identify_zero_importance(task = 'classification', n_iterations = 10, early_stopping = False)
fs.identify_low_importance(cumulative_importance=0.99)
selected_training_set = fs.remove(methods = ['zero_importance']) # 可选'collinear', 'zero_importance', 'low_importance'
remain_features = list(selected_training_set) # 查看保留的特征
# removed_features  = fs.check_removal() # 查看移除的特征
# print(removed_features)

### 保存或者加载特征
#### 保存筛选出的特征
#save_load_remainFeature("save", save_load_path, remain_features)
'''
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

# wifi定位
## 数据准备
TP_path = "./data/WiFi/TP.csv"
RP_path = "./data/WiFi/RP.csv"

TP_df = pd.read_csv(TP_path)
RP_df = pd.read_csv(RP_path)

RP_rssi = RP_df.iloc[:, 3:].values # RP的RSSI
RP_position = RP_df.iloc[:, 0:3].values # RP位置
TP_rssi = TP_df.iloc[:, 3:].values # TP的RSSI


## 实例化模型并定位
wifi = wifi.Model(RP_rssi)
wifi_predict, wifi_accuracy = wifi.wknn_strong_signal_reg(RP_rssi, RP_position, TP_rssi, realTrace)
#wifi.show_3D_trace(wifi_predict, real_trace=realTrace)
print(f'WiFi平均定位误差:{wifi_accuracy} m')

# pdr定位
## 数据准备
walking_data_file = test_path + '/pdr_data.csv'
df_walking = pd.read_csv(walking_data_file)

linear = df_walking[[col for col in df_walking.columns if 'linear' in col]].values[startidx:]
gravity = df_walking[[col for col in df_walking.columns if 'gravity' in col]].values[startidx:]
rotation = df_walking[[col for col in df_walking.columns if 'rotation' in col]].values[startidx:]
gyro = df_walking[[col for col in df_walking.columns if 'gyr' in col]].values[startidx:]

## 数据平滑
linear = smooth_data(linear)
gravity = smooth_data(gravity)
rotation = smooth_data(rotation)
gyro = smooth_data(gyro)

pdr = pdr.Model(linear, gravity, rotation, gyro, CNNParameterPath=CNNParameter_Path)
X_pdr, Y_pdr, Z_pdr, strides, angle, delt_h = pdr.pdr_position(frequency=freq, walkType='normal', \
                        offset = 0,initPosition=(0, 0, 0),\
                        fuse_oritation = False, \
                        predictPattern=predictions_svc, m_WindowWide=window_wide)
pdr.show_trace(frequency=freq, walkType='normal', initPosition=(0, 0, 0),\
                predictPattern=predictions_svc, m_WindowWide=window_wide,\
                real_trace=realTrace)

x = np.array(X_pdr).reshape(-1, 1)
y = np.array(Y_pdr).reshape(-1, 1)
z = np.array(Z_pdr).reshape(-1, 1)
pdr_predict = np.concatenate((x, y, z), axis=1)

mean_pdr_error = ave_accuracy(pdr_predict, realTrace)

print(f"PDR平均定位误差:{mean_pdr_error} m")

# EKF融合定位
## 准备数据
X_real, Y_real, Z_real = realTrace[:,0], realTrace[:,1], realTrace[:,2]
X_wifi, Y_wifi, Z_wifi = wifi_predict[:,0], wifi_predict[:,1], wifi_predict[:,2]
L = strides #步长
## 超参数设置
sigma_wifi = 15
sigma_pdr = 0
sigma_yaw = 15/360
sigma_h = 1
fusion = fusion.Model()
theta_counter = -1
def state_conv(parameters_arr):
    global theta_counter
    theta_counter += 1
    x = parameters_arr[0]
    y = parameters_arr[1]
    z = parameters_arr[2]
    theta = parameters_arr[3]
    X = x+L[theta_counter]*np.sin(theta)
    Y = y+L[theta_counter]*np.cos(theta)
    Z = z+delt_h[theta_counter]
    Angle = angle[theta_counter]
    return X, Y, Z, Angle

# 观测矩阵(zk)，目前不考虑起始点（设定为0，0），因此wifi数组长度比实际位置长度少1
observation_states = []
for i in range(len(angle)):
    x = X_wifi[i]
    y = Y_wifi[i]
    z = Z_wifi[i]
    observation_states.append(np.matrix([
        [x], [y], [z], [L[i]], [angle[i]]
    ]))

#状态矩阵（xk)
transition_states = []
for k, v in enumerate(angle):
    x = X_pdr[k]
    y = Y_pdr[k]
    z = Z_pdr[k]
    theta = angle[k]
    V = np.matrix([[x],[y],[z],[theta]])
    transition_states.append(V) #数组里保存着4*1的矩阵，对应4个状态（pdr输出的x,y,z与方向角）

# 状态协方差矩阵（初始状态不是非常重要，经过迭代会逼近真实状态）
P = np.matrix([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1]])
# 观测矩阵
H = np.matrix([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 1]])
# 状态转移协方差矩阵
Q = np.matrix([[sigma_pdr**2, 0, 0, 0],
               [0, sigma_pdr**2, 0, 0],
               [0, 0, sigma_h**2, 0],
               [0, 0, 0, sigma_yaw**2]])
# 观测噪声方差
R = np.matrix([[sigma_wifi**2, 0, 0, 0, 0],
               [0, sigma_wifi**2, 0, 0, 0],
               [0, 0, sigma_wifi**2, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 0, 0, sigma_yaw**2]])
#状态转移雅各比行列式
def jacobF_func(i):
    return np.matrix([[1, 0, 0, L[i]*np.cos(angle[i])],
                      [0, 1, 0, -L[i]*np.sin(angle[i])],
                      [0, 0, 0, 0],
                      [0, 0, 0, 1]])

S = fusion.ekf3d(
    transition_states = transition_states # 状态矩阵
   ,observation_states = observation_states # 观测数组
   ,transition_func = state_conv # 状态预测函数（传入参数为数组格式，该数组包含了用到的状态转换遇到的数据）
   ,jacobF_func = jacobF_func # 一阶线性的状态转换公式
   ,initial_state_covariance = P
   ,observation_matrices = H
   ,transition_covariance = Q
   ,observation_covariance = R
)

X_ekf = []
Y_ekf = []
Z_ekf = []

for v in S:
    X_ekf.append(-v[0, 0])
    Y_ekf.append(v[1, 0])
    Z_ekf.append(v[2, 0])

ekf_predict = np.concatenate((np.array(X_ekf).reshape(len(X_ekf), 1),
                              np.array(Y_ekf).reshape(len(Y_ekf), 1),
                              np.array(Z_ekf).reshape(len(Z_ekf), 1)), axis=1)

accuracy = fusion.ave_accuracy(ekf_predict, realTrace)


mean_ekf_error = ave_accuracy(ekf_predict, realTrace)
print(f'ekf 平均定位误差:{mean_ekf_error} m')
fusion.show_3D_trace(ekf_predict, real_trace=realTrace)

## 画所有定位轨迹图

config = {
        "font.family":'Times New Roman',  # 设置字体类型
            #     "mathtext.fontset":'stix',
        }
rcParams.update(config)
fig = plt.figure()
ax = fig.gca(projection='3d')
plt.grid()
ax.plot(X_real, Y_real, Z_real, 'o-', label='Real tracks')
ax.plot(X_wifi, Y_wifi, Z_wifi, 'r.', label='WiFi positioning')
ax.plot(X_pdr, Y_pdr, Z_pdr, 'o-', label='PDR positioning')
ax.plot(X_ekf, Y_ekf, Z_ekf, 'o-', label='EKF positioning')

ax.set_xlabel('X', fontsize=20)#设置横纵坐标标签
ax.set_ylabel('Y', fontsize=20)
plt.xticks(fontsize=18) #设置坐标轴刻度大小
plt.yticks(fontsize=18)
plt.legend(fontsize = 20)
#plt.show()
#plt.savefig('./Figure/all_location_trace.jpg',format='jpg',bbox_inches = 'tight',dpi=300)