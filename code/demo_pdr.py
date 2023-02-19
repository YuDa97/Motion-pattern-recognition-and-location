# 数据平滑
from recognition.build_dataset import smooth_data
# pdr模型
import location.pdr as pdr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics.pairwise import cosine_similarity
import os

freq = 25
init_time = 3
init_data = freq * init_time # 初始化数据
path = 'D:/motion sense/Motion-pattern-recognition/data/TestData'
walking_data_file = path + '/exp1/pdr_data.csv'
real_trace_file = path + '/test_coordinate.csv'
#fuse_yaw_file = 'D:/动态定位/PDR+WIFI+EKF/SensorFusion-master/source-code/fuse_yaw.csv'
real_trace = pd.read_csv(real_trace_file).loc[:, 'x':'z'].values # 真实轨迹

df_walking = pd.read_csv(walking_data_file)

## 获得线性加速度、重力加速度、姿态仰角的numpy.ndarray数据,去掉初始化数据点
linear = df_walking[[col for col in df_walking.columns if 'linear' in col]].values[init_data:]
gravity = df_walking[[col for col in df_walking.columns if 'gravity' in col]].values[init_data:]
rotation = df_walking[[col for col in df_walking.columns if 'rotation' in col]].values[init_data:]
## 数据平滑
linear = smooth_data(linear)
gravity = smooth_data(gravity)
rotation = smooth_data(rotation)

pdr = pdr.Model(linear, gravity, rotation)

# # Demo1：显示垂直方向合加速度与步伐波峰分布
# # frequency：数据采集频率
# # walkType：行走方式（normal为正常走路模式，abnormal为做融合定位实验时走路模式）
pdr.show_steps(frequency=25, walkType='normal')

# # Demo2：显示数据在一定范围内的分布情况，用来判断静止数据呈现高斯分布
# # 传入参数为静止状态x（y或z）轴线性加速度
# acc_x = linear[:, 0]
# acc_y = linear[:, 1]
#acc_z = linear[:,2]
#pdr.show_gaussian(acc_z, False)

# # Demo3：显示三轴线性加速度分布情况
#pdr.show_data('rotation')
#pdr.show_data('linear')
#pdr.show_data('gravity') #是否能用重力加速度变化作步数估计？

# # Demo4：获取步伐信息
# # 返回值steps为字典类型，index为样本序号，acceleration为步伐加速度峰值，v_acceleration为谷值
steps = pdr.step_counter(frequency=25, walkType='normal')
print('steps:', len(steps))
#stride = pdr.step_stride # 步长推算函数实例化
# # 计算步长推算的平均误差
#accuracy = []
#for v in steps:
#    a = v['acceleration']
#    b = v['v_acceleration']
    #print(stride(a, b, k=0.37))
#    accuracy.append(
#        np.abs(stride(a, b, k=0.37)-0.6)#步长固定是0.6m, k值对结果有较大影响。
#    )
#square_sum = 0
#for v in accuracy:
#    square_sum += v*v
#acc_mean = (square_sum/len(steps))**(1/2)
#print("mean: %f" % acc_mean) # 平均误差 
#print("min: %f" % np.min(accuracy)) # 最小误差
#print("max: %f" % np.max(accuracy)) # 最大误差
#print("sum: %f" % np.sum(accuracy)) # 累积误差

# # Demo5：获取每个采样点的航向角
#theta = pdr.step_heading()[:10]
#temp = theta[0]
#for i,v in enumerate(theta):
#    theta[i] = np.abs(v-temp)*360/(2*np.pi)
#    print(theta[i])
#print("mean: %f" % np.mean(theta))

# 获取每一步的航向角
#x,y,stride_length, angle = pdr.pdr_position(frequency=70, walkType="abnormal",  initPosition=(7,12))
#angle_list = []
#for i in angle:
#    angle_list.append(i*180/np.pi)
#print("每一步的航向角为：",angle_list)

# Demo6：显示PDR预测轨迹
# 注意：PDR不清楚初始位置与初始航向角
#pdr.show_trace(frequency=70, walkType='normal')
#test_x_pdr, test_y_pdr, test_strides, test_angle = pdr.pdr_position(frequency=70, walkType='fusion', offset=0, initPosition=(6.64,11.07))
#test_x_pdr = np.array(test_x_pdr).reshape(-1,1)
#test_y_pdr = np.array(test_y_pdr).reshape(-1,1)
#pdr_test_result = np.concatenate((test_x_pdr,test_y_pdr),axis = 1)
#pdr_error = np.sqrt(np.sum((pdr_test_result - real_trace)**2, 1))*0.62
#pdr_mean_accuracy = np.sqrt(np.mean(np.sum((pdr_test_result - real_trace)**2, 1)))*0.62
#print("PDR平均误差:{:.2f}".format(pdr_mean_accuracy))
#pdr.show_trace(frequency=25, walkType='normal',\
#                offset=-np.pi/2, initPosition=(0,0,0),\
#                real_trace=real_trace,)


