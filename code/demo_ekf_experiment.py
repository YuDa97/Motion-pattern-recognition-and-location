#%%
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import location.pdr as pdr
import location.wifi as wifi
import location.fusion as fusion

'''
真实实验
注意假如有n个状态，那么就有n-1次状态转换
'''

frequency = 70 # 数据采集频率
sigma_wifi = 4.1
sigma_pdr = .3
sigma_yaw = 15/360
# 初始状态
X = np.matrix('6.64; 11.07; 0')
# X = np.matrix('2; 2; 0') # 对初始状态进行验证

path = 'e:/动态定位/PDR+WIFI+EKF/location-master/wlData'
real_trace_file = path + '/PDRTest/RealTrace_train2.csv'
fuse_yaw_file = 'E:/动态定位/PDR+WIFI+EKF/SensorFusion-master/source-code/fuse_yaw.csv'#融合航向角
walking_data_file = path + '/PDRTest/train_k30_2.csv'
fingerprint_path = path + '/FingerprintMap'

df_walking = pd.read_csv(walking_data_file) # 实验数据
real_trace = pd.read_csv(real_trace_file).values # 真实轨迹
df_fuse_yaw = pd.read_csv(fuse_yaw_file)
# 主要特征参数
rssi = df_walking[[col for col in df_walking.columns if 'rssi' in col]].values[300:]
linear = df_walking[[col for col in df_walking.columns if 'linear' in col]].values[300:]
gravity = df_walking[[col for col in df_walking.columns if 'gravity' in col]].values[300:]
rotation = df_walking[[col for col in df_walking.columns if 'rotation' in col]].values[300:]
fuse_yaw_test = df_fuse_yaw[[col for col in df_fuse_yaw.columns if 'k30_2' in col]].values[300:] 
pdr = pdr.Model(linear, gravity, rotation, fuse_yaw_test, min_acc=0.17)
wifi = wifi.Model(rssi)
fusion = fusion.Model()

# 指纹数据
fingerprint_rssi, fingerprint_position = wifi.create_fingerprint(fingerprint_path)

# 找到峰值出的rssi值
steps = pdr.step_counter(frequency=frequency, walkType='fusion')
print('steps:', len(steps))
result = fingerprint_rssi[0].reshape(1, rssi.shape[1])
for k, v in enumerate(steps):
    index = v['index']
    value = rssi[index]
    value = value.reshape(1, len(value))
    result = np.concatenate((result,value),axis=0)

# wknn算法
predict, accuracy = wifi.wknn_reg(fingerprint_rssi, fingerprint_position, result, real_trace)
print('wknn accuracy:', accuracy, 'm')
predict = np.array(predict)
predict[0][0] = 6.64
predict[0][1] = 11.07
init_x = X[0, 0]
init_y = X[1, 0]
init_angle = X[2, 0]
x_pdr, y_pdr, strides, angle = pdr.pdr_position(frequency=frequency, walkType='fusion', offset=init_angle, initPosition=(init_x, init_y))

# ekf
X_real = real_trace[:,0]
Y_real = real_trace[:,1]
X_wifi = predict[:,0]
Y_wifi = predict[:,1]
X_pdr = x_pdr
Y_pdr = y_pdr
L = strides #步长

theta_counter = -1
def state_conv(parameters_arr):
    global theta_counter
    theta_counter = theta_counter+1
    x = parameters_arr[0]
    y = parameters_arr[1]
    theta = parameters_arr[2]
    return x+L[theta_counter]*np.sin(theta), y+L[theta_counter]*np.cos(theta), angle[theta_counter]

# 观测矩阵(zk)，目前不考虑起始点（设定为0，0），因此wifi数组长度比实际位置长度少1
observation_states = []
for i in range(len(angle)):
    x = X_wifi[i]
    y = Y_wifi[i]
    observation_states.append(np.matrix([
        [x], [y], [L[i]], [angle[i]]
    ]))

#状态矩阵（xk)
transition_states = []
for k, v in enumerate(angle):
    x = X_pdr[k]
    y = Y_pdr[k]
    theta = angle[k]
    V = np.matrix([[x],[y],[theta]])
    transition_states.append(V) #数组里保存着3*1的矩阵，对应3个状态（pdr输出的x,y与方向角）

# 状态协方差矩阵（初始状态不是非常重要，经过迭代会逼近真实状态）
P = np.matrix([[1, 0, 0],
               [0, 1, 0],
               [0, 0, 1]])
# 观测矩阵
H = np.matrix([[1, 0, 0],
               [0, 1, 0],
               [0, 0, 0],
               [0, 0, 1]])
# 状态转移协方差矩阵
Q = np.matrix([[sigma_pdr**2, 0, 0],
               [0, sigma_pdr**2, 0],
               [0, 0, sigma_yaw**2]])
# 观测噪声方差
R = np.matrix([[sigma_wifi**2, 0, 0, 0],
               [0, sigma_wifi**2, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, sigma_yaw**2]])
#状态转移雅各比行列式
def jacobF_func(i):
    return np.matrix([[1, 0, L[i]*np.cos(angle[i])],
                      [0, 1, -L[i]*np.sin(angle[i])],
                      [0, 0, 1]])

S = fusion.ekf2d(
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

for v in S:
    X_ekf.append(v[0, 0])
    Y_ekf.append(v[1, 0])

ekf_predict = np.concatenate((np.array(X_ekf).reshape(len(X_ekf), 1),
                              np.array(Y_ekf).reshape(len(Y_ekf), 1)), axis=1)
accuracy = fusion.square_accuracy(ekf_predict, real_trace)

print('ekf accuracy:', accuracy, 'm')
ekf_error = np.sqrt(np.sum((ekf_predict - real_trace)**2, 1))*0.62
x = X_ekf
y = Y_ekf
#%%
#将数据写入excel
ekf_error_df = pd.DataFrame(ekf_error)
writer = pd.ExcelWriter('e:/动态定位/PDR+WIFI+EKF/location-master/results/ekf_error.xlsx')  #创建名称为lstm调参报告的excel表格
ekf_error_df.to_excel(writer,'page_1',float_format='%.5f')  #float_format 控制精度，将data_df写到hhh表格的第一页中。若多个文件，可以在page_2中写入
writer.save() 
#%%
#plot
from matplotlib import rcParams
config = {
        "font.family":'Times New Roman',  # 设置字体类型
            #     "mathtext.fontset":'stix',
        }
rcParams.update(config)
plt.figure(figsize=(15,8),dpi=300)
#for k in range(0, len(x)):
#    plt.annotate(k, xy=(x[k], y[k]), xytext=(x[k]+0.1,y[k]+0.1))

plt.grid()
plt.plot(X_real, Y_real, 'o-', label='Real tracks')
plt.plot(X_wifi, Y_wifi, 'r.', label='WiFi positioning')
plt.plot(X_pdr, Y_pdr, 'o-', label='PDR positioning')
plt.plot(X_ekf, Y_ekf, 'o-', label='EKF positioning')
ax=plt.gca()
ax.set_xlabel('X', fontsize=20)#设置横纵坐标标签
ax.set_ylabel('Y', fontsize=20)
plt.xticks(fontsize=18) #设置坐标轴刻度大小
plt.yticks(fontsize=18)
plt.legend(fontsize = 20)
#plt.savefig('E:/动态定位/PDR+WIFI+EKF/location-master/Figures/ekf.jpg',format='jpg',bbox_inches = 'tight',dpi=300)