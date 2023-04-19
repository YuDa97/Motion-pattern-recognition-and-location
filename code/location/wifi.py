'''
1.Model参数列表（1个参数）：
WIFI信号强度矩阵

2.Model参数类型：
numpy.ndarray
'''

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import ensemble
import matplotlib.ticker as ticker
from scipy.stats import norm
from sklearn import neighbors, svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
import heapq
from sklearn.preprocessing import MinMaxScaler
# import plda
from sklearn.decomposition import PCA
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class Model(object):
    def __init__(self, rssi):
        self.rssi = rssi
    def pca(self, x_train, x_test, perc_of_var):
        '''
        Preforms PCA and keeps perc_of_var percent of variance 
    
        Parameters : x_train      : (DataFrame) Training Dataset
                 x_test       : (DataFrame) Test Dataset
                 perc_of_var  : (float) percent of variance from PCA
    
        Returns    : x_train      : (DataFrame) Training Dataset
                    x_test       : (DataFrame) Test Dataset
        '''   
        dim_red = PCA(n_components=perc_of_var, svd_solver='full')
        x_train = dim_red.fit_transform(x_train)
        x_test = dim_red.transform(x_test)
    
        return x_train, x_test

    def create_fingerprint(self, path, for_plda=False, num_of_rss=None):
        
        if for_plda is False:
            RSSI = None
            X = None
            Y = None
            # path为采集的指纹数据目录，每个坐标为一个文件，文件的命名格式为：x_y
            directory = os.walk(path)  
            for _, _, file_list in directory:
                for file_name in file_list:
                    position = file_name.split('.')[0].split('-') # 获取文件记录的坐标
                    x = np.array([[int(position[0])]])
                    y = np.array([[int(position[1])]])
                    df = pd.read_csv(path + "/" + file_name)
                    columns = [col for col in df.columns if 'rssi' in col]
                    rssi = df[columns].values
                    rssi = rssi[300:] # 视前300行数据为无效数据
                    rssi_mean = np.mean(rssi, axis=0).reshape(1, rssi.shape[1])
                    if RSSI is None:
                        RSSI = rssi_mean#均值滤波
                        X = x
                        Y = y
                    else:
                        RSSI = np.concatenate((RSSI,rssi_mean), axis=0)
                        X = np.concatenate((X,x), axis=0)
                        Y = np.concatenate((Y,y), axis=0)
            fingerprint = np.concatenate((RSSI, X, Y), axis=1)
            fingerprint = pd.DataFrame(fingerprint, index=None, columns = columns+['x', 'y'])
            rssi = fingerprint[[col for col in fingerprint.columns if 'rssi' in col]].values
            position = fingerprint[['x', 'y']].values
            return rssi, position
        elif for_plda is True:
            RSSI = np.array([[None]*num_of_rss])
            
            X = np.array([[None]])
            Y = np.array([[None]])
            C = np.array([[None]])
            directory = os.walk(path)  
            for _, _, file_list in directory:
                for file_name in file_list:
                    position = file_name.split('.')[0].split('-') # 获取文件记录的坐标
                    x = np.array([[int(position[0])]])
                    y = np.array([[int(position[1])]])
                    c = np.array([[int(position[2])]])#记录参考点所属类别
                    df = pd.read_csv(path + "/" + file_name)
                    columns = [col for col in df.columns if 'rssi' in col]
                    rssi = df[columns].dropna().values
                    mean_1 = np.array([np.mean(rssi[300:1300,:],0)])
                    mean_2 = np.array([np.mean(rssi[1300:2300,:],0)])
                    mean_3 = np.array([np.mean(rssi[2300:,:],0)])
                    rssi = np.concatenate((mean_1,mean_2,mean_3),axis=0)
                    
                    class_label = c
                    x_label = x
                    y_label = y
                    for i in range(rssi.shape[0]-1):
                        class_label = np.concatenate((class_label,c))
                        x_label = np.concatenate((x_label,x))
                        y_label = np.concatenate((y_label,y))
                    class_label.reshape(-1,1)
                    x_label.reshape(-1,1)
                    y_label.reshape(-1,1)
                    C = np.concatenate((C,class_label), axis=0)
                    RSSI = np.concatenate((RSSI,rssi), axis=0)
                    X = np.concatenate((X,x_label), axis=0)
                    Y = np.concatenate((Y,y_label), axis=0)
                    
            fingerprint = np.concatenate((RSSI, X, Y, C), axis=1)
            fingerprint = pd.DataFrame(fingerprint, index=None, columns = columns+['x', 'y','class'])
            fingerprint=fingerprint.dropna(axis=0)
            
            rssi = fingerprint[[col for col in fingerprint.columns if 'rssi' in col]].values
            position_df = fingerprint[['x', 'y']].drop_duplicates()
            position = position_df.values
            offline_label = fingerprint['class'].values
            return rssi, position, offline_label

    # 标准差
    def square_accuracy(self, predictions, labels):
        accuracy = np.sqrt(np.mean(np.sum((predictions - labels)**2, 1)))*0.62
        return round(accuracy, 3)

    def ave_accuracy(self, predictions, labels):
        accuracy = np.mean((np.sum((predictions - labels)**2, 1))**0.5)*0.62
        return round(accuracy, 3)
    
    
    #添加区域限制的回归算法
    def ml_limited_reg(self, type, offline_rss, offline_location, online_rss, online_location):
        if type == 'knn':
            k = 3
            ml_reg = neighbors.KNeighborsRegressor(n_neighbors=k, weights='uniform', metric='euclidean')
        elif type == 'rf':
            ml_reg = RandomForestRegressor(n_estimators=10)

        init_x = 7
        init_y = 12
        predict = np.array([[init_x, init_y]])
        limited_rss = None
        limited_location = None
        offset = 2 # m

        for k, v in enumerate(online_rss):
            if k == 0:
                continue
            for v1, v2 in zip(offline_rss, offline_location):
                if (v2[0] >= init_x-offset and v2[0] <= init_x+offset) and (v2[1] >= init_y-offset and v2[1] <= init_y+offset):
                    v1 = v1.reshape(1, v1.size)
                    v2 = v2.reshape(1, v2.size)
                    if limited_rss is None:
                        limited_rss = v1
                        limited_location = v2
                    else:
                        limited_rss = np.concatenate((limited_rss, v1), axis=0)
                        limited_location = np.concatenate((limited_location, v2), axis=0)
            v = v.reshape(1, v.size)
            predict_point = ml_reg.fit(limited_rss, limited_location).predict(v)
            predict = np.concatenate((predict, predict_point), axis=0)
            init_x = predict_point[0][0]
            init_y = predict_point[0][1]
            limited_rss = None
            limited_location = None
        
        accuracy = self.ave_accuracy(predict, online_location)
        return predict, accuracy

    # 选取信号最强的num个rssi作为匹配
    def wknn_strong_signal_reg(self, offline_rss, offline_location, online_rss, online_location):
        num = 8
        k = 3
        rssi_length = offline_rss.shape[1]
        knn_reg = neighbors.KNeighborsRegressor(n_neighbors=k, weights='distance', metric='euclidean')

        limited_location = None

        for rssi in online_rss:
            keys = np.argsort(rssi)[(rssi_length - num):]
            # keys = np.argsort(rssi)[:num]
            rssi = rssi.reshape(1, rssi_length)
            limited_online_rssi = rssi[:,keys] # from small to big
            limited_offline_rssi = offline_rss[:,keys]
            predict_point = knn_reg.fit(limited_offline_rssi, offline_location).predict(limited_online_rssi)
            if limited_location is None:
                limited_location = predict_point
            else:
                limited_location = np.concatenate((limited_location, predict_point), axis=0)

        predict = limited_location
        accuracy = self.ave_accuracy(predict, online_location)
        return predict, accuracy

    # knn regression
    def knn_reg(self, offline_rss, offline_location, online_rss, online_location):
        k = 3
        knn_reg = neighbors.KNeighborsRegressor(n_neighbors=k, weights='uniform', metric='euclidean')
        predict = knn_reg.fit(offline_rss, offline_location).predict(online_rss)
        accuracy = self.square_accuracy(predict, online_location)
        return predict, accuracy
    
    # wknn regression
    def wknn_reg(self, offline_rss, offline_location, online_rss, online_location):
        k = 3
        wknn_reg = neighbors.KNeighborsRegressor(n_neighbors=k, weights='distance', metric='euclidean')
        predict = wknn_reg.fit(offline_rss, offline_location).predict(online_rss)
        accuracy = self.ave_accuracy(predict, online_location)
        return predict, accuracy
    
    # 支持向量机
    def svm_reg(self, offline_rss, offline_location, online_rss, online_location):
        clf_x = svm.SVR(C=1000, gamma=0.01)
        clf_y = svm.SVR(C=1000, gamma=0.01)
        clf_x.fit(offline_rss, offline_location[:, 0])
        clf_y.fit(offline_rss, offline_location[:, 1])
        x = clf_x.predict(online_rss)
        y = clf_y.predict(online_rss)
        predict = np.column_stack((x, y))
        accuracy = self.ave_accuracy(predict, online_location)
        return predict, accuracy
    
    # 随机森林
    def rf_reg(self, offline_rss, offline_location, online_rss, online_location):
        estimator = RandomForestRegressor(n_estimators=150)
        estimator.fit(offline_rss, offline_location)
        predict = estimator.predict(online_rss)
        accuracy = self.ave_accuracy(predict, online_location)
        return predict, accuracy

    # 梯度提升
    def dbdt(self, offline_rss, offline_location, online_rss, online_location):
        clf = MultiOutputRegressor(ensemble.GradientBoostingRegressor(n_estimators=100, max_depth=10))
        clf.fit(offline_rss, offline_location)
        predict = clf.predict(online_rss)
        accuracy = self.ave_accuracy(predict, online_location)
        return predict, accuracy
    
    # 多层感知机
    def nn(self, offline_rss, offline_location, online_rss, online_location):
        clf = MLPRegressor(hidden_layer_sizes=(100, 100))
        clf.fit(offline_rss, offline_location)
        predict = clf.predict(online_rss)
        accuracy = self.ave_accuracy(predict, online_location)
        return predict, accuracy

    #K-PLDA
    '''
    def plda_reg(self, offline_rss, offline_label, offline_location, online_rss, online_location):#使用plda来预测待估计点坐标函数
        clf = plda.Classifier()
        offline_rss, online_rss = self.pca(offline_rss, online_rss,0.95)

        clf.fit_model(offline_rss, offline_label)


        rec_predictions, rec_log_p_predictions = clf.predict(online_rss)#计算待测点与参考点相似的对数概率（rec_log_p_predictions)


        scaler = MinMaxScaler( )
        scaler.fit(rec_log_p_predictions)
        scaler.data_max_
        rec_log_p_predictions=scaler.transform(rec_log_p_predictions)#将对数似然概率归一化
 
        rec_predictions = rec_log_p_predictions
    
        #对类别概率进行筛选
        rows,cols=rec_log_p_predictions.shape#读取X的行列数
        #取前k个最大的概率
   
        k = 10
        lar_list = []
        for i in range(rows):#提取每一行中最大的k个值
            every_row = rec_predictions[i]
            k_largest = heapq.nlargest(k,every_row)
            lar_list.append(k_largest)

        for i in range(rows):#将非k值清零
            for j in range(cols):
                if rec_predictions[i][j] not in lar_list[i]:
                    rec_predictions[i][j] = 0
  
        #将概率归一化
        rec_predictions_sum = 1 / np.sum(rec_predictions,axis = 1) #取求和后的倒数
        rec_predictions_sum = (rec_predictions_sum.reshape(rec_predictions.shape[0],1)).repeat(rec_predictions.shape[1],axis=1)#将它变形成与rec_predictions一样

        c = rec_predictions * rec_predictions_sum  #储存归一化后的概率
    
        predict = np.dot(c,offline_location) #输出预测坐标，注意这里矩阵乘法顺序不能搞反
        accuracy = self.ave_accuracy(predict, online_location)
        return predict, accuracy
        
#添加区域限制的PLDA算法
    def limited_plda_reg(self, offline_rss, offline_location, label, online_rss, online_location, init_location, limited_location=None):
        clf = plda.Classifier()
        offline_rss, online_rss = self.pca(offline_rss, online_rss,0.95)
        
        limited_rss = None
        predict = init_location
        offset = 3.5 # m
        offline_rss = offline_rss.reshape(-1,3,offline_rss.shape[1])
        i = limited_location
        label =  label.reshape(-1,3,1)
        for k, v in enumerate(online_rss):#从第二个点开始添加区域限制
            if k == 0:
                continue
            
            for v1, v2, v3 in zip(offline_rss, offline_location, label):
                if (v2[0] >= i[k-1][0]-offset and v2[0] <= i[k-1][0]+offset) and (v2[1] >= i[k-1][1]-offset and v2[1] <= i[k-1][1]+offset):
                    #v1 = v1.reshape(1, v1.size)
                    v2 = v2.reshape(1, v2.size)
                    if limited_rss is None:
                        limited_rss = v1
                        limited_RP = v2
                        limited_label = v3
                    else:
                        limited_rss = np.concatenate((limited_rss, v1), axis=0)
                        limited_RP = np.concatenate((limited_RP, v2), axis=0)
                        limited_label = np.concatenate((limited_label, v3), axis=0)
            limited_label = limited_label.reshape(1,-1).squeeze()
            
            clf.fit_model(limited_rss, limited_label)
            rec_predictions, rec_log_p_predictions = clf.predict(v)
            scaler = MinMaxScaler( )
            rec_log_p_predictions = rec_log_p_predictions.reshape(-1,1)
            scaler.fit(rec_log_p_predictions)
            scaler.data_max_
            rec_log_p_predictions=scaler.transform(rec_log_p_predictions)#将对数似然概率归一化

            rec_predictions = rec_log_p_predictions
                
            #将概率归一化
            rec_predictions_sum = 1 / np.sum(rec_predictions,axis = 0) #取求和后的倒数
               
            rec_predictions_sum = rec_predictions_sum.repeat(rec_predictions.shape[1],axis=0)#将它变形成与rec_predictions一样
                
                
            c = rec_predictions_sum * rec_predictions #储存归一化后的概率
            pre_location = np.dot(c.T,limited_RP)
            predict = np.concatenate((predict, pre_location), axis=0)
            limited_rss = None
            limited_RP = None
        
        accuracy = self.square_accuracy(predict, online_location)
        return predict, accuracy
        '''
#添加区域限制的WKNN
    def limited_wknn(self, offline_rss, offline_location, online_rss, online_location, init_location, limited_location):
        k = 4
        limited_rss = None
        predict = init_location
        ml_reg = neighbors.KNeighborsRegressor(n_neighbors=k, weights='distance', metric='euclidean')
        offset = 7 # m
        i = limited_location
        for k, v in enumerate(online_rss):
            if k == 0:
                continue
            for v1, v2 in zip(offline_rss, offline_location):
                if (v2[0] >= i[k-1][0]-offset and v2[0] <= i[k-1][0]+offset) and (v2[1] >= i[k-1][1]-offset and v2[1] <= i[k-1][1]+offset):
                    v1 = v1.reshape(1, v1.size)
                    v2 = v2.reshape(1, v2.size)
                    if limited_rss is None:
                        limited_rss = v1
                        limited_location = v2
                    else:
                        limited_rss = np.concatenate((limited_rss, v1), axis=0)
                        limited_location = np.concatenate((limited_location, v2), axis=0)
            v = v.reshape(1, v.size)
            predict_point = ml_reg.fit(limited_rss, limited_location).predict(v)
            predict = np.concatenate((predict, predict_point), axis=0)
            limited_rss = None
            limited_location = None
        
        accuracy = self.ave_accuracy(predict, online_location)
        return predict, accuracy



    '''
        data为np.array类型
    '''
    def determineGaussian(self, data, merge, interval=1, wipeRange=300):
        offset = wipeRange
        data = data[offset:]

        minValue = np.min(data)
        maxValue = np.max(data)
        meanValue = np.mean(data)

        length = math.ceil((maxValue-minValue)/interval)
        counterArr = length * [0]
        valueRange = length * [0]

        textstr = '\n'.join((
                r'$max=%.2f$' % (maxValue, ),
                r'$min=%.2f$' % (minValue, ),
                r'$mean=%.2f$' % (meanValue, )))

        if merge==True:
            # 区间分段样本点
            result = []
            temp_data = data[0]
            for i in range(0, len(data)):
                if temp_data == data[i]:
                    continue
                else:
                    result.append(temp_data)
                    temp_data = data[i]
            data = result

        for index in range(len(counterArr)):
            valueRange[index] = minValue + interval*index

        for value in data:
            key = int((value - minValue) / interval)
            if key >=0 and key <length:
                counterArr[key] += 1
        
        if merge==True:
            print('Wi-Fi Scan Times:', len(data))

        probability = np.array(counterArr) / np.sum(counterArr)
        normal_mean = np.mean(data)
        normal_sigma = np.std(data)
        normal_x = np.linspace(minValue, maxValue, 100)
        normal_y = norm.pdf(normal_x, normal_mean, normal_sigma)
        normal_y = normal_y * np.max(probability) / np.max(normal_y)

        _, ax = plt.subplots()

        # Be sure to only pick integer tick locations.
        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_major_locator(ticker.MaxNLocator(integer=True))

        ax.bar(valueRange, probability, label='distribution')
        ax.plot(normal_x, normal_y, 'r-', label='fitting')
        plt.xlabel('rssi value')
        plt.ylabel('probability')
        plt.title('信号强度数据的高斯拟合')
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
        plt.legend()
        plt.show()
    
    def rssi_fluctuation(self, merge, wipeRange=300):
        # wipeRange=300表示前300行数据中包含了无效数据，可以直接去除
        offset = wipeRange
        rssi = self.rssi[offset:]
        rows = rssi.shape[0]
        columns = rssi.shape[1]
        lines = [0]*(columns+1)
        labels = [0]*(columns+1)

        filename = ''

        if merge == False:
            for i in range(0, columns):
                lines[i], = plt.plot(range(0, rows), rssi[:, i])
                labels[i] = 'rssi' + str(i+1)
            
            plt.title('指纹库文件'+filename+'数据波动情况')
            plt.legend(handles=lines, labels=labels, loc='best')
            plt.xlabel('样本点数目/个')
            plt.ylabel('WiFi信号强度/dBm')
            plt.show()

        elif merge == True:
            # 采集周期（以一定样本点数目为一个周期）
            indexs = []
            results = []
            
            # 区间分段样本点
            for i in range(0, columns):
                counter = 0
                intervals = []
                result = []
                temp_rssi = rssi[0, i]
                for j in range(0, rows):
                    if temp_rssi == rssi[j, i]:
                        counter = counter +1
                    else:
                        intervals.append(counter)
                        result.append(temp_rssi)
                        temp_rssi = rssi[j, i]
                indexs.append(intervals)
                results.append(result)
                intervals = []
            
            # 确定最小长度
            length = 0
            for i in range(0, columns):
                if length==0:
                    length = len(results[i])
                else:
                    if len(results[i]) < length:
                        length = len(results[i])
            
            # 显示图像
            for i in range(0, columns):
                lines[i], = plt.plot(range(0, length), results[i][:length])
                labels[i] = 'rssi' + str(i+1)
            
            plt.title('指纹库文件'+filename+'数据波动情况')
            plt.legend(handles=lines, labels=labels, loc='best')
            plt.xlabel('WiFi扫描次数/次')
            plt.ylabel('WiFi信号强度/dBm')
            plt.xticks(range(0, length, int(length/5))) # 保证刻度为整数
            plt.show()

    # 显示运动轨迹图
    def show_trace(self, predict_trace, **kw):
        from matplotlib import rcParams
        config = {
            "font.family":'Times New Roman',  # 设置字体类型
            #     "mathtext.fontset":'stix',
                }
        rcParams.update(config)
        plt.figure(figsize=(15,8),dpi=300)
        plt.grid()
        handles = []
        labels = []
        if 'real_trace' in kw:
            real_trace = kw['real_trace'].T
            trace_x = real_trace[0]
            trace_y = real_trace[1]
            l1, = plt.plot(trace_x, trace_y, 'o-')
            handles.append(l1)
            labels.append('Real tracks')
            #for k in range(0, len(trace_x)):
            #    plt.annotate(k, xy=(trace_x[k], trace_y[k]), xytext=(trace_x[k]+0.1,trace_y[k]+0.1), color='green')

        predict = predict_trace.T
        x = predict[0]
        y = predict[1]

        #for k in range(0, len(x)):
        #    plt.annotate(k, xy=(x[k], y[k]), xytext=(x[k]+0.1,y[k]+0.1))
        ax=plt.gca()
        ax.set_xlabel('X', fontsize=20)#设置横纵坐标标签
        ax.set_ylabel('Y', fontsize=20)
        l2, = plt.plot(x, y, 'o')
        handles.append(l2)
        labels.append('WiFi predicting')
        plt.scatter(x, y, c ='r')
        plt.legend(handles=handles ,labels=labels, loc='best', fontsize = 20)
        plt.xticks(fontsize=18) #设置坐标轴刻度大小
        plt.yticks(fontsize=18)
        plt.show()
        #plt.savefig('E:/动态定位/PDR+WIFI+EKF/location-master/Figures/wifi_limited.jpg',format='jpg',bbox_inches = 'tight',dpi=300)