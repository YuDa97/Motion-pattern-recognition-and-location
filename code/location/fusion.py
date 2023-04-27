'''
1.Model参数类型：
numpy.ndarray
'''

import numpy as np
import random
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class Model(object):
    def __init__(self):
        pass

    # 标准差
    def square_accuracy(self, predictions, labels):
        accuracy = np.sqrt(np.mean(np.sum((predictions - labels)**2, 1)))
        return round(accuracy, 3)

    def ave_accuracy(self, predictions, labels):
        accuracy = np.mean(np.sqrt(np.sum((predictions - labels)**2, 1)))
        return round(accuracy, 3)
        
    def ekf2d(
        self
       ,transition_states
       ,observation_states
       ,transition_func
       ,jacobF_func
       ,initial_state_covariance
       ,observation_matrices
       ,transition_covariance
       ,observation_covariance
        ):

        conv_length = len(transition_states)-1

        # 状态参数个数
        initial_state = transition_states[0] 
        state_parameters_num = initial_state.shape[0] #4
        # 单个状态参数数组，形式为：[[x],[y],[z],[theta]]
        state_parameters = [0]*state_parameters_num
        temp = []
        for i in range(state_parameters_num):
            for v in transition_states:
                temp.append(v[i, 0])
            state_parameters[i] = temp
            temp = []

        # 获取单个观测参数
        observation_parameters_num = observation_states[0].shape[0]
        observation_parameters = [0]*observation_parameters_num
        for i in range(observation_parameters_num):
            for v in observation_states:
                temp.append(v[i, 0])
            observation_parameters[i] = temp
            temp = []

        S = [] # 融合估计位置
        k_list = [] #卡尔曼增益系数
        X = initial_state # 初始状态
        # X = np.matrix('2; 2; 0') # 对初始状态进行验证
        S.append(X)
        P = initial_state_covariance # 状态协方差矩阵（初始状态不是非常重要，经过迭代会逼近真实状态）
        Q = transition_covariance # 状态转移协方差矩阵
        H = observation_matrices # 观测矩阵
        R = observation_covariance # 观测噪声方差

        # LType-03 data
        # sigma_wifi = np.array([12.62, 12.39, 13.7, 24.38, 26.74, 25.63, 23.93, 33.14, 18.53, 28.68, 34.44, 30.8, 32.67, 32.22, 41.48, 31.78, 27.58, 26.5, 34.85, 25.11, 31.97, 25.74, 23.13, 25.55, 29.25, 31.05, 33.26, 33.16, 31.39, 33.95])
        # sigma_wifi = sigma_wifi/10
        # sigma_observation = [R]
        # for v in sigma_wifi:
        #     sigma_observation.append(
        #         np.matrix([[v**2, 0, 0, 0],
        #                     [0, v**2, 0, 0],
        #                     [0, 0, 0, 0],
        #                     [0, 0, 0, 0.1**2]])
        #     )
        
        for i in range(conv_length):
            # 状态预测
            state_values = [X[k, 0] for k in range(state_parameters_num)]
            
            new_state_values = transition_func(state_values)# 状态转移方程：x+L[theta_counter]*np.sin(theta), y+L[theta_counter]*np.cos(theta), angle[theta_counter]
            X_ = np.matrix([[new_state_values[k]] for k in range(state_parameters_num)])#预测

            # 一阶线性化后的状态矩阵
            # if testing is 1:
            #     R = sigma_observation[i]
            F = jacobF_func(i)
            P_ = F * P * F.T + Q#预测
            K = P_ * H.T * np.linalg.pinv(H * P_ * H.T + R) #卡尔曼增益系数
            Z = np.matrix([[observation_parameters[k][i+1]] for k in range(observation_parameters_num)]) # 将wifi定位作为新息
            X = X_ + K * (Z - H * X_)#对pdr进行修正
            P = (np.eye(4) - K * H) * P_#更新
            k_list.append(K)
            S.append(X)
        
        return S
    
    def show_3D_trace(self, predict_trace, **kw):
        '''
        显示三维运动轨迹图
        '''
        from matplotlib import rcParams
        config = {
            "font.family":'Times New Roman',  # 设置字体类型
            #     "mathtext.fontset":'stix',
                }
        rcParams.update(config)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        plt.grid()
        handles = []
        labels = []
        if 'real_trace' in kw:
            real_trace = kw['real_trace'].T
            trace_x = real_trace[0]
            trace_y = real_trace[1]
            trace_z = real_trace[2]
            l1, = ax.plot(trace_x, trace_y, trace_z,'o')
            handles.append(l1)
            labels.append('Real tracks')
            #for k in range(0, len(trace_x)):
            #    plt.annotate(k, xy=(trace_x[k], trace_y[k]), xytext=(trace_x[k]+0.1,trace_y[k]+0.1), color='green')

        predict = predict_trace.T
        x = predict[0]
        y = predict[1]
        z = predict[2]

        #for k in range(0, len(x)):
        #    plt.annotate(k, xy=(x[k], y[k]), xytext=(x[k]+0.1,y[k]+0.1))
        
        ax.set_xlabel('X', fontsize=20)#设置横纵坐标标签
        ax.set_ylabel('Y', fontsize=20)
        ax.set_zlabel('Z', fontsize=20)
        l2, = ax.plot(x, y, z, '-o')
        handles.append(l2)
        labels.append('EKF predicting')
        #plt.scatter(x, y, c ='r')
        plt.legend(handles=handles ,labels=labels, loc='best', fontsize = 20)
        plt.xticks(fontsize=18) #设置坐标轴刻度大小
        plt.yticks(fontsize=18)
        plt.show()
        #plt.savefig('./Figure/ekf_location_trace.jpg',format='jpg',bbox_inches = 'tight',dpi=300)