from cProfile import label
import numpy as np
import pandas as pd
from recognition.extract_feature import GetFeature
import scipy.signal
import pywt
import matplotlib.pyplot as plt
from matplotlib import rcParams
from adtk.detector import LevelShiftAD
from adtk.visualization import plot
import os


def smooth_data(data, window_length=9, k=3):
    '''
    平滑数据
    data:代表曲线点坐标(x,y)中的y值数组
    window_length:窗口长度,该值需为正奇整数。window_length的值越小,曲线越贴近真实曲线;window_length值越大,平滑效果越厉害
    k值:polyorder为对窗口内的数据点进行k阶多项式拟合,k的值需要小于window_length。
        k值对曲线的平滑作用: k值越大,曲线越贴近真实曲线; k值越小,曲线平滑越厉害。
        另外,当k值较大时,受窗口长度限制,拟合会出现问题,高频曲线会变成直线。
    '''
    smooth_data = scipy.signal.savgol_filter(data, window_length, k, axis=0)
    return smooth_data

def cwt_data(data, wavename='cgau8'):
    '''
    对数据进行连续小波变换
    trn_data: ndarry(2-Dim), trainData
    tst_data: ndarry(2-Dim), testData
    '''
    sampling_rate = 1024
    totalscal = 9 # 频率尺度
    fc = pywt.central_frequency(wavename)
    cparam = 2 * fc * totalscal
    scales = cparam / np.arange (totalscal , 0 , -1)
    cwtmatr ,frequencies = pywt.cwt(data, scales, wavename, 1.0 / sampling_rate)
    
    return cwtmatr, frequencies

def creat_pattern_dataset(file):
    raw_data = pd.read_csv(file)
    raw_data = raw_data.drop(['time(ms)', 'RV.heading'], axis=1)
    raw_data = raw_data.values
    smo_data = smooth_data(raw_data)
    
    return raw_data, smo_data

def cal_res_acc(smo_acc):
    '''
    计算合加速度
    输入:
    smo_acc: ndarry(2-Dim), 平滑后的三轴加速度
    输出:
    res_acc: ndarry(2-Dim), 第一维是合加速度大小, 第二维是与xoy平面的夹角(弧度)
    '''
    res_acc = np.zeros((0, 2)) # 第一列为合加速度的模，第二列为合加速度与xoy平面夹角
    for i in smo_acc:
        x_vector = np.array([i[0], 0, 0])
        y_vector = np.array([0, i[1], 0])
        z_vector = np.array([0, 0, i[2]])
        temp_vector = x_vector + y_vector + z_vector
        temp_vector_value = np.sqrt(np.sum(temp_vector**2)) #计算模 
        temp_vector_direction = np.arcsin(abs(i[2])/temp_vector_value) # 计算合加速度与xoy平面夹角
        res_acc_temp = np.array([[temp_vector_value, temp_vector_direction]])
        res_acc = np.append(res_acc, res_acc_temp, axis=0)
    return res_acc[:, 0] 
    
def anomalies_detect(data, windowWide, showFigure=None):
    '''
    突变检测
    input: data->ndarry
           windowWide->int
           showFigure->bool: 是否查看突变检测图像
    output: anomalies_index: 发生突变的窗口下标索引
    '''
    data = pd.DataFrame(data)
    data.index = pd.to_datetime(data.index)
    level_shift_ad = LevelShiftAD(c=1.05, side='both', window=windowWide)
    anomalies = level_shift_ad.fit_detect(data)
    if showFigure:
        plot(data, anomaly=anomalies, anomaly_color="red", anomaly_tag="marker")
        plt.show()
    n = anomalies.shape[0]
    window_index, anomalies_index = 0, []
    for i in range(0, n, windowWide):
        window_data = anomalies.iloc[i:i+windowWide, 0]
        if window_data.any():
            anomalies_index.append(window_index)
        window_index += 1
    return  anomalies_index
    
def plot_time_serise(raw_data, smo_data, cwtmatr, frequencies, res_acc, data_type):
    raw_feature_index = {'acc': raw_data[:, 0:3], 'acclin': raw_data[:, 3:6], 
                     'gra': raw_data[:, 6:9], 'gyro': raw_data[:, 9:12],
                     'rv': raw_data[:, 12:-1]}
    smo_feature_index = {'acc': smo_data[:, 0:3], 'acclin': smo_data[:, 3:6], 
                    'gra': smo_data[:, 6:9], 'gyro': smo_data[:, 9:12],
                    'rv': smo_data[:, 12:-1]}
    '''
    freq_feature_index = {'acc': freq_data[:, 0:3], 'acclin': freq_data[:, 3:6], 
                'gra': freq_data[:, 6:9], 'gyro': freq_data[:, 9:12],
                'rv': freq_data[:, 12:-1]}
    '''
    config = {
    "font.family":'Times New Roman',  # 设置字体类型
    #     "mathtext.fontset":'stix',
    }
    raw_y = raw_feature_index[data_type]
    smo_y = smo_feature_index[data_type]
    # freq_y = freq_feature_index[data_type]
    rcParams.update(config)
    plt.figure()
    # 时域图，未平滑
    plt.subplot(3, 2, 1)
    ax=plt.gca()
    ax.set_xlabel('T', fontsize=20)#设置横纵坐标标签
    ax.set_ylabel('Values', fontsize=20)
    x = np.arange(1, raw_data.shape[0]+1) # x轴坐标

    plt.plot(x, raw_y[:, 0], label='x axix')
    plt.plot(x, raw_y[:, 1], label='y axis')
    plt.plot(x, raw_y[:, 2], label='z axis')
    plt.xticks(fontsize=18) #设置坐标轴刻度大小
    plt.yticks(fontsize=18)
    plt.legend()
    
    # 时域图，平滑
    plt.subplot(3, 2, 2)
    ax=plt.gca()
    ax.set_xlabel('T', fontsize=20)#设置横纵坐标标签
    ax.set_ylabel('Values', fontsize=20)
    plt.plot(x, smo_y[:, 0], label='x axix')
    plt.plot(x, smo_y[:, 1], label='y axis')
    plt.plot(x, smo_y[:, 2], label='z axis')
    plt.xticks(fontsize=18) #设置坐标轴刻度大小
    plt.yticks(fontsize=18)
    plt.legend()
    
    # 合加速度时域图
    plt.subplot(3, 2, 3)
    ax=plt.gca()
    ax.set_xlabel('T', fontsize=20)#设置横纵坐标标签
    ax.set_ylabel('Values', fontsize=20)
    plt.plot(x, res_acc, label='res acc values')
    # plt.plot(x, res_acc[:, 1], label='res acc directions') # 合加速度方向
    plt.xticks(fontsize=18) #设置坐标轴刻度大小
    plt.yticks(fontsize=18)
    plt.legend()
    
    # 合加速度频域图
    plt.subplot(3, 2, 4)
    ax=plt.gca()
    ax.set_xlabel('T', fontsize=20)#设置横纵坐标标签
    ax.set_ylabel('Frequencies', fontsize=20)
    plt.contourf(x, frequencies, abs(cwtmatr))
    plt.subplots_adjust(hspace=0.4)
    plt.xticks(fontsize=18) #设置坐标轴刻度大小
    plt.yticks(fontsize=18)
    plt.legend()

    # 取某一频率下合加速度频域:
    plt.subplot(3, 2, 5)
    ax=plt.gca()
    ax.set_xlabel('T', fontsize=20)#设置横纵坐标标签
    ax.set_ylabel('Frequencies', fontsize=20)
    plt.plot(x, abs(cwtmatr[8]), label='56.9Hz')
    plt.xticks(fontsize=18) #设置坐标轴刻度大小
    plt.yticks(fontsize=18)
    plt.legend()
    plt.show()


def slide_window(data, wide, label, time_label=None, freq=25, startidx=75):
    '''
    给data加窗,并提取每个窗口的特征值组建训练集,此处为不重叠滑动窗口
    input: For building training set:
           data: ndarry(2-Dim);
           wide: 窗口宽度(int)
           label: 训练集数据标签(int)
           For building testing set:
           time_label: 测试集时间序列标签(object)
           freq: 采样频率
           startidx: 数据分析的起始下标
    output: features->ndarry(2-Dim)
            featurn_name->ndarry(1-Dim)
    '''
    n = data.shape[0]
    feature_name = np.zeros((0))
    if time_label is not None:
        m, j = time_label.shape[0], 0
        shift_idx = np.zeros((0)) # 保存运动状态转变时的下标
        startidx = int(time_label[0][0] * freq)
    for i in range(0, n, wide):
        win_feature = np.zeros((0)) # 一个窗口内的特征
        win_acc = data[i:i+wide, :] # 三轴加速度
        res_acc = cal_res_acc(win_acc) # 合加速度
        feature = GetFeature(res_acc, win_acc)
        for attr, value in feature.__dict__.items(): 
            if '_acc' in attr: # unverified!!
                continue
            else:
                win_feature = np.append(win_feature, value) # 组合特征
        if type(label) == int:
            label_feature = np.append(win_feature, label).reshape(1, -1) # 给训练数据附上标签
        ## 当输入的label是字典, 构建测试集
        else: 
            
            if j < m:
                part = time_label[j] # 对于测试集中第j段运动模式
                if j + 1 < m:
                    next_part = time_label[j+1] # 第J段运动模式
                start, end = int(part[0] * freq) - startidx, int(part[1]*freq) - startidx
                label_code = label[part[2]]
                if i >= start and i+wide <= end:
                    label_feature = np.append(win_feature, label_code).reshape(1, -1)
                    if i+wide == end:
                        j += 1
                elif i >= start and i+wide > end:
                    if i+wide-end <= int(wide/2):
                        label_feature = np.append(win_feature, label[part[2]]).reshape(1, -1) # 转换处标签延续上一时刻
                    else:
                         label_feature = np.append(win_feature, label[next_part[2]]).reshape(1, -1)# 转换处标签延续下一时刻
                    #label_feature = np.append(win_feature, -1).reshape(1, -1) # 如果窗口处于运动状态转换处，标记为-1
                    j += 1
                    
                
        if np.isnan(label_feature).sum():
            continue 
        if i == 0:
            dataset = label_feature
        else:
            if (type(label) != int) and (i >= start and i+wide > end):
                dataset = np.concatenate((dataset, label_feature), axis=0)
                shift_idx = np.append(shift_idx, dataset.shape[0] - 1)
            else:
                dataset = np.concatenate((dataset, label_feature), axis=0)
    for attr, value in feature.__dict__.items(): 
        if 'acc' in attr: # unverified!!
            continue
        else:
            feature_name = np.append(feature_name, attr)    
    return dataset, feature_name
            

def creat_training_set(path, label_coding, startidx, wide, training_dimention):
    '''
    构建训练集
    input: path->str: 训练集数据路径
           label_coding->dic: 标签与编号映射
           startidx->int: 从第几个数据开始, 滤除初始化数据
           wide->int: 数据窗宽度
           training_dimention->int: 训练集列数(特征数加标签)
    output: training_set->ndarry: 训练集, 2-Dim
            feature_name->ndarry: 特征名称, 1-Dim
    '''
    training_set = np.zeros((0, training_dimention))
    for state_name in os.listdir(path): # 每个运动状态
        motion_path = path + '/' + state_name
        label = label_coding[state_name]
        training_for_file = np.zeros((0, training_dimention))
        for data_file in os.listdir(motion_path): # 每个运动状态中的文件
            file_path = motion_path + '/' + data_file + '/' + 'AccelerometerLinear.csv'
            raw_acc = pd.read_csv(file_path).loc[:, 'X':'Z'].values
            raw_acc = raw_acc[startidx:]
            smo_acc = smooth_data(raw_acc) # 平滑
            temp_training, feature_name  = slide_window(smo_acc, wide, label)
            training_for_file = np.concatenate((training_for_file, temp_training), axis=0)
        training_set = np.concatenate((training_set, training_for_file), axis=0)
    return training_set, feature_name

def creat_testing_set(exp_path, label_coding, startidx, freq, wide, data_dimention):
    '''
    构建测试集
    input: exp_path->str: 测试集数据路径,需要具体到第几个实验
           label_coding->dic: 标签与编号映射
           startidx->int: 从第几个数据开始, 滤除初始化数据
           wide->int: 数据窗宽度
           data_dimention->int: 测试集列数(特征数加标签)
    output: training_set->ndarry: 测试集, 2-Dim

    '''
    testing_set = np.zeros((0, data_dimention))
    file_path = exp_path + '/' + 'AccelerometerLinear.csv'
    time_label_path = exp_path + '/' + 'time.csv'
    raw_acc = pd.read_csv(file_path).loc[:, 'X':'Z'].values
    time_label = pd.read_csv(time_label_path).iloc[1:, 1:4].values
    raw_acc = raw_acc[startidx:]
    smo_acc = smooth_data(raw_acc) # 平滑
    temp_tesing, feature_name  = slide_window(smo_acc, wide, label_coding, time_label, freq, startidx)
    testing_set = np.concatenate((testing_set, temp_tesing), axis=0)
    return testing_set

def creat_anomalies_detect_dataset(exp_path):
    '''
    创建用于突变点检测的测试集
    input: test_path->str: 数据集所在路径
    output: res_acc->ndarry(1-D): 合加速度
    '''
    file_path = exp_path + '/' + 'AccelerometerLinear.csv'
    raw_acc = pd.read_csv(file_path).loc[:, 'X':'Z'].values
    smo_acc = smooth_data(raw_acc) 
    res_acc = cal_res_acc(smo_acc)
    return res_acc

if __name__ == "__main__":
    
    train_path = '/home/yuda/Motion-pattern-recognition/data/TrainData'
    test_path = '/home/yuda/Motion-pattern-recognition/data/TestData/exp1'
    freq = 25 # 数据采样频率是25Hz
    label_coding = {'stand': 0, 'walk': 1, 'up': 2, 'down': 3}
    feature_num = 44
    training_dimention = feature_num + 1
    startidx = 75 # 舍掉前75个点
    window_wide = int(1.5 * freq) # 滑动窗口宽度
    training_set, feature_name = creat_training_set(train_path, label_coding, startidx, window_wide, training_dimention)
    test_set = creat_testing_set(test_path, label_coding, startidx, freq, window_wide, training_dimention)
    

    print(training_set.shape)
    # print(feature_name)
    print(test_set.shape)
    
    '''
    ## 平滑滤波&突变检测调参
    file_path = '/home/yuda/Motion-pattern-recognition/data/demo/WL2DW0944_0860/motion_data.csv'
    freq = 25
    window_wide = int(1.5 * freq)
    raw_data, smo_data = creat_pattern_dataset(file_path)
    res_acc = cal_res_acc(smo_data[:, 0:3]) # 计算合加速度
    cwtmatr ,frequencies = cwt_data(res_acc)
    anomalies_detect(abs(cwtmatr[8]), window_wide, showFigure=True)
    plot_time_serise(raw_data, smo_data, cwtmatr ,frequencies, res_acc, 'acc')
    '''