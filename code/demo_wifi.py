import location.pdr as pdr
import location.wifi as wifi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

path = 'e:/动态定位/PDR+WIFI+EKF/location-master/wlData'

real_trace_file = path + '/PDRTest/RealTrace_train7.csv'
walking_data_file = path + '/PDRTest/train_k30_7.csv'
fingerprint_path = path + '/FingerprintMap'

df_walking = pd.read_csv(walking_data_file) # 实验数据
real_trace = pd.read_csv(real_trace_file).values # 真实轨迹

# 主要特征参数
rssi = df_walking[[col for col in df_walking.columns if 'rssi' in col]].values
linear = df_walking[[col for col in df_walking.columns if 'linear' in col]].values
gravity = df_walking[[col for col in df_walking.columns if 'gravity' in col]].values
rotation = df_walking[[col for col in df_walking.columns if 'rotation' in col]].values

pdr = pdr.Model(linear, gravity, rotation,min_acc=0.19)
wifi = wifi.Model(rssi)

# 指纹数据
fingerprint_rssi, fingerprint_position = wifi.create_fingerprint(fingerprint_path)
#K-PLDA指纹库
#fingerprint_rssi, fingerprint_position, offline_label = wifi.create_fingerprint(fingerprint_path, for_plda=True, num_of_rss = 20)

# 找到峰值处的rssi值, for traditional algorithm
steps = pdr.step_counter(frequency=70, walkType='abnormal')
print('steps:', len(steps))
result = fingerprint_rssi[0].reshape(1, rssi.shape[1])
for k, v in enumerate(steps):
    index = v['index']
    value = rssi[index]
    value = value.reshape(1, len(value))
    result = np.concatenate((result,value),axis=0)

# # knn算法
#predict, accuracy = wifi.knn_reg(fingerprint_rssi, fingerprint_position, result, real_trace)
#print('knn accuracy:', accuracy, 'm')
#wifi.show_trace(predict, real_trace=real_trace)

# # wknn算法
predict, accuracy = wifi.wknn_reg(fingerprint_rssi, fingerprint_position, result, real_trace)
predict[0][0] = 6.64
predict[0][1] = 11.07
print('wknn accuracy:', accuracy, 'm')
wifi.show_trace(predict, real_trace=real_trace)

# 添加区域限制的knn回归
#predict, accuracy = wifi.ml_limited_reg('knn', fingerprint_rssi, fingerprint_position, result, real_trace)
#print('knn_limited accuracy:', accuracy, 'm')

# svm算法
#predict, accuracy = wifi.svm_reg(fingerprint_rssi, fingerprint_position, result, real_trace)
#print('svm accuracy:', accuracy, 'm')
#wifi.show_trace(predict, real_trace=real_trace)


# rf算法
#predict, accuracy = wifi.rf_reg(fingerprint_rssi, fingerprint_position, result, real_trace)
#print('rf accuracy:', accuracy, 'm')
#wifi.show_trace(predict, real_trace=real_trace)

# 添加区域限制rf的rf算法
#predict, accuracy = wifi.ml_limited_reg('rf', fingerprint_rssi, fingerprint_position, result, real_trace)
#print('rf_limited accuracy:', accuracy, 'm')
#wifi.show_trace(predict, real_trace=real_trace)
# gdbt算法
#predict, accuracy = wifi.dbdt(fingerprint_rssi, fingerprint_position, result, real_trace)
#print('gdbt accuracy:', accuracy, 'm')
#wifi.show_trace(predict, real_trace=real_trace)
# 多层感知机
#predict, accuracy = wifi.nn(fingerprint_rssi, fingerprint_position, result, real_trace)
#print('nn accuracy:', accuracy, 'm')
#wifi.show_trace(predict, real_trace=real_trace)

#K-PLDA
#predict, accuracy = wifi.plda_reg(fingerprint_rssi, offline_label, fingerprint_position, result, real_trace)
#print('K-PLDA accuracy:', accuracy, 'm')
#wifi.show_trace(predict, real_trace=real_trace)

#加了区域限制的K-PLDA
#limited_location = np.array([[7,12],[6,12]])
#init_location = np.array([[7,12]])
#predict, accuracy = wifi.limited_plda_reg(fingerprint_rssi, fingerprint_position, offline_label, result[0:2], real_trace[0:2], limited_location=limited_location, init_location = init_location)
#wifi.show_trace(predict, real_trace=real_trace)
#添加区域限制的WKNN
#limited_location = np.array([[7,12],[6,12]])
#init_location = np.array([[7,12]])
#predict, accuracy = wifi.limited_wknn(fingerprint_rssi, fingerprint_position, result[0:2], real_trace[0:2], limited_location=limited_location, init_location = init_location)
#wifi.show_trace(predict, real_trace=real_trace)