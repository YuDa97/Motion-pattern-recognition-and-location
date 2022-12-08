from cProfile import label
import numpy as np
import pandas as pd
from extract_feature import GetFeature
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
    
def anomalies_detect(data):
    data = pd.DataFrame(data)
    data.index = pd.to_datetime(data.index)
    level_shift_ad = LevelShiftAD(c=6.0, side='both', window=5)
    anomalies = level_shift_ad.fit_detect(data)
    plot(data, anomaly=anomalies, anomaly_color="red", anomaly_tag="marker")
    
    
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
    plt.plot(x, res_acc[:, 0], label='res acc values')
    plt.plot(x, res_acc[:, 1], label='res acc directions')
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


def slide_window(data, wide, label):
    '''
    给data加窗,并提取每个窗口的特征值组建训练集,此处为不重叠滑动窗口
    input: data: ndarry(2-Dim);
           wide: 窗口宽度(int)
    output: features ndarry(2-Dim)
    '''
    n = data.shape[0]
    feature_name = np.zeros((0))
    for i in range(0, n, wide):
        win_feature = np.zeros((0)) # 一个窗口内的特征
        win_acc = data[i:i+wide, :] # 三轴加速度
        res_acc = cal_res_acc(win_acc) # 合加速度
        feature = GetFeature(res_acc, win_acc)
        for attr, value in feature.__dict__.items(): 
            if 'acc' in attr: # unverified!!
                continue
            else:
                win_feature = np.append(win_feature, value) # 组合特征
        label_feature = np.append(win_feature, label).reshape(1, -1) # 给数据附上标签
        if np.isnan(label_feature).sum():
            continue 
        if i == 0:
            training_set = label_feature.reshape(1, -1)
        else:
            training_set = np.concatenate((training_set, label_feature), axis=0)
    for attr, value in feature.__dict__.items(): 
        if 'acc' in attr: # unverified!!
            continue
        else:
            feature_name = np.append(feature_name, attr)    
    return training_set, feature_name
            

def creat_training_set(path, label_coding, startidx, wide, training_dimention):
    '''
    构建训练集
    input: path->str: 训练集数据路径
           label_coding->dic: 标签与编号映射
           startidx->int: 从第几个数据开始, 滤除初始化数据
           wide->int: 数据窗宽度
           training_dimention->int: 训练集列数(特征数加标签)
    output: training_set->ndarry: 训练集, 2-Dim

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

def creat_testing_set(path):
    pass




if __name__ == "__main__":
    path = 'E:/PDR/motion-pattern/data/TrainData'
    freq = 25 # 数据采样频率是25Hz
    label_coding = {'stand': 0, 'walk': 1, 'up': 2, 'down': 3}
    feature_num = 8 
    training_dimention = feature_num + 1
    startidx = 70 # 舍掉前70个点
    window_wide = int(1.5 * freq) # 滑动窗口宽度
    training_set, feature_name = creat_training_set(path, label_coding, startidx, window_wide, training_dimention)
    print(training_set.shape)
    print(feature_name)

    '''
    ## Here we set parameter to build labeld time-series from dataset of "(A)DeviceMotion_data"
    num_features = 12 # attitude(roll, pitch, yaw); gravity(x, y, z); rotationRate(x, y, z); userAcceleration(x,y,z)
    num_act_labels = 6 # dws, ups, wlk, jog, sit, std
    num_gen_labels = 1 # 0/1(female/male)
    label_codes = {"dws":num_features, "ups":num_features+1, "wlk":num_features+2, "jog":num_features+3, "sit":num_features+4, "std":num_features+5}
    trial_codes = {"dws":[1,2,11], "ups":[3,4,12], "wlk":[7,8,15], "jog":[9,16], "sit":[5,13], "std":[6,14]}    
    ## Calling 'creat_time_series()' to build time-series
    print("--> Building Training and Test Datasets...")
    train_ts, test_ts = creat_time_series(num_features, num_act_labels, num_gen_labels, label_codes, trial_codes)
    print("--> Shape of Training Time-Seires:", train_ts.shape)
    print("--> Shape of Test Time-Series:", test_ts.shape)
    

    ## 运动模式识别demo
    file_path = 'E:/motion sense/motion-pattern/data/WL2DW0944_0860/motion_data.csv'
    raw_data, smo_data = creat_pattern_dataset(file_path)
    res_acc = cal_res_acc(smo_data[:, 0:3]) # 计算合加速度
    cwtmatr ,frequencies = cwt_data(res_acc[:, 0])
    anomalies_detect(abs(cwtmatr[8]))
    plot_time_serise(raw_data, smo_data, cwtmatr ,frequencies, res_acc, 'acc')
    '''