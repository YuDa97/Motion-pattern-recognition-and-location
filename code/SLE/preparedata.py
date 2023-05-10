import os
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn import preprocessing
from scipy import integrate
from sklearn.model_selection import train_test_split

class LoadData():
    '''
    获取上下楼的加速度,重力加速度和陀螺仪数据
    低通滤波
    通过检步算法提取相应加速度和陀螺仪数据
    '''
    def __init__(self, path, freq) -> None:
        
        # 数据文件路径
        self.path = path
        
        # 采样频率
        self.freq = freq
        
        # 数据初始化长度
        self.startIdx = 75
        
        # 真实步长
        self.true_length = None

        # 步伐信息
        self.step_info = None
        
        # 原始数据 加速度计三轴,陀螺仪三轴,重力加速度三轴,真实步长
        self.raw_dataset = np.zeros((0, 10))
        
        # 滤波后的数据        
        self.filt_dataset = None
        self.filt_acc = None
        self.filt_gyro = None
        self.filt_grav = None
        self.a_vertical = None
        
        
    def read_data(self):
        '''
        读取上下楼加速度,重力加速度和陀螺仪数据,保存到 self.dataset属性中
        '''
        for step_nums in os.listdir(self.path): # 不同步长
            if step_nums == "onestep":
                label = 0.34
            elif step_nums == "twostep":
                label = 0.68
            data_path = self.path + '/' + step_nums
            data_for_file = np.zeros((0, 10)) # 加速度三轴，陀螺仪三轴，重力加速度三轴, 真实步长
            for data_file in os.listdir(data_path): # 每个不同步长中的文件
                acc_file_path = data_path + '/' + data_file + '/' + 'AccelerometerLinear.csv'
                gyro_file_path = data_path + '/' + data_file + '/' + 'Gyroscope.csv'
                grav_file_path = data_path + '/' + data_file + '/' + 'Gravity.csv'
                
                raw_acc = pd.read_csv(acc_file_path).loc[:, 'X':'Z'].values
                raw_acc = raw_acc[self.startIdx:]
                raw_gyro = pd.read_csv(gyro_file_path).loc[:, 'X':'Z'].values
                raw_gyro = raw_gyro[self.startIdx:]
                raw_grav = pd.read_csv(grav_file_path).loc[:, 'X':'Z'].values
                raw_grav = raw_grav[self.startIdx:]
                
                row_arry = [raw_acc.shape[0], raw_gyro.shape[0], raw_grav.shape[0]]
                # 统一维度
                min_row = min(row_arry) 
                if raw_acc.shape[0] > min_row:
                    raw_acc = raw_acc[:min_row, :]
                if raw_gyro.shape[0] > min_row:
                    raw_gyro = raw_gyro[:min_row, :]
                if raw_grav.shape[0] > min_row:
                    raw_grav = raw_grav[:min_row, :]
                label_array = np.full(shape=(min_row,1),fill_value=label) 
                
                acc_gyro_gra_l = np.concatenate((raw_acc, raw_gyro, raw_grav, label_array), axis=1)
                data_for_file = np.concatenate((data_for_file, acc_gyro_gra_l), axis=0)
            self.raw_dataset = np.concatenate((self.raw_dataset, data_for_file), axis=0)
        self.true_length = self.raw_dataset[:, -1] 
        
    def filter_data(self):
        '''
        巴特沃斯低通滤波
        '''
        b, a = signal.butter( 4,  0.25,  'lowpass') # 配置滤波器4表示滤波器的阶数,Wn=2*截止频率/采样频率
        self.filt_dataset = signal.filtfilt(b, a, self.raw_dataset[:, 0:-1], axis=0)
        self.filt_acc = self.filt_dataset[:, 0:3]
        self.filt_gyro = self.filt_dataset[:, 3:6]
        self.filt_grav = self.filt_dataset[:, 6:9]
        
        
    def coordinate_conversion(self):
        '''
        计算垂直方向加速度
        '''
        gravity = self.filt_grav
        linear = self.filt_acc

        # g_x = gravity[:, 0]
        g_y = gravity[:, 1]
        g_z = gravity[:, 2]

        # linear_x = linear[:, 0]
        linear_y = linear[:, 1]
        linear_z = linear[:, 2]
        
        theta = np.arctan(np.abs(g_z/g_y))

        # 得到垂直方向加速度（除去g）
        self.a_vertical = linear_y*np.cos(theta) + linear_z*np.sin(theta)

        return self.a_vertical
    
    def step_counter(self, frequency=25, walkType='normal', **kw):
        '''
        步数检测函数

        walkType取值:
        normal: 正常行走模式
        abnormal: 融合定位行走模式(每一步行走间隔大于1s)

        返回值：
        steps
        字典型数组,每个字典保存了峰值位置(index)与该点的垂直方向加速度值(acceleration)
        '''
        offset = 0.4
        g = 0.96
        self.coordinate_conversion()
        slide = int(frequency * offset) # 滑动窗口长度
    
        # 行人加速度阈值
        min_acceleration = 0.6 * g #0.576
        max_acceleration = 6.7 * g   # 6.43
        valleyWin_scale = 37 # 谷值窗口宽度
        # 峰值间隔(s)
        min_interval = 0.4 * frequency
        # max_interval = 1
        # 计算步数
        steps = []
        peak = {'index': 0, 'acceleration': 0, \
                'v_index': 0, 'v_acceleration': 0, \
                'm_pattern': -2} # v_index：谷值索引，v_acceleration：谷值, m_pattern: 运动模式
        #peaks, properities = signal.find_peaks(a_vertical, height=min_acceleration, distance=min_interval) #调用scipy检测峰值

        
        # 以宽度为slide的滑动窗检测谷值,选择在峰值后加窗
        # 条件1:峰值在min_acceleration~max_acceleration之间
        for i, v in enumerate(self.a_vertical):
            if v >= peak['acceleration'] and v >= min_acceleration and v <= max_acceleration:
                peak['acceleration'] = v
                peak['index'] = i
            if i%slide == 0 and peak['index'] != 0:
                valleyWin_start = peak['index'] 
                valleyWin = self.a_vertical[valleyWin_start:valleyWin_start+valleyWin_scale]
                peak['v_acceleration'] = np.min(valleyWin)
                peak['v_index'] = int(np.argwhere(valleyWin == np.min(valleyWin))[0]) + valleyWin_start
                if 'motionPattern' in kw:
                    pattern_index = int(i/kw['motionPatternWindowWide'])
                    peak['m_pattern'] = kw['motionPattern'][pattern_index]
                steps.append(peak)
                peak = {'index': 0, 'acceleration': 0, 'v_index': 0, 'v_acceleration': 0, 'm_pattern': -2}
        
        # 条件2：两个峰值之前间隔至少大于0.4s*frequency
        # del使用的时候，一般采用先记录再删除的原则
        if len(steps)>0:
            lastStep = steps[0]
            dirty_points = []
            for key, step_dict in enumerate(steps):
                # print(step_dict['index'])
                if key == 0:
                    continue
                if step_dict['index']-lastStep['index'] < min_interval:
                    # print('last:', lastStep['index'], 'this:', step_dict['index'])
                    if step_dict['acceleration'] <= lastStep['acceleration']:#如果当前峰值小于上一峰值
                        dirty_points.append(key)#删去当前峰值
                    else:
                        lastStep = step_dict
                        dirty_points.append(key-1)
                else:
                    lastStep = step_dict
            
            counter = 0 # 记录删除数量，作为偏差值，删除以后下标会移动
            for key in dirty_points:
                del steps[key-counter]
                counter = counter + 1
        self.step_info = steps
        return steps 
    
    def build_dataset(self):
        '''
        创建数据集
        加速度三轴,陀螺仪三轴,步频
        先检步,以峰值为中心前后各加w/2个窗口
        '''
         
        self.read_data()
        self.filter_data()
        steps = self.step_counter()
        # print("步数总数:", len(steps))
        w = 14 # 窗口宽度
        dataset = np.zeros((0, 7, w+1))
        label = np.zeros((0))
        for i, v in enumerate(steps):
            if i == 0:
                continue
            last_step = steps[i-1] # 上一步
            if v['index'] > 7 and v['index'] + int(w/2) + 1 < self.filt_dataset.shape[0]:
                w_start = v['index'] - int(w/2) # 窗口起始索引
                w_end = v['index'] + int(w/2) + 1 # 窗口结束索引
                
                w_acc = self.filt_acc[w_start: w_end,:]
                w_gyro = self.filt_gyro[w_start: w_end,:]
                t = (v['index'] - last_step['index']) * 1/self.freq # 步频
                t_array = np.full(shape=(w_acc.shape[0],1),fill_value=t) # 转换成数组
                
                # 获取当前步真实步长
                w_length = self.true_length[v['index']]
                w_data = np.concatenate((w_acc, w_gyro, t_array), axis=1).T # 7*15, 方便标准化和输入到CNN
                w_data = preprocessing.normalize(w_data, norm='l2') # 标准化
                w_data = np.array([w_data])
                dataset = np.append(dataset, w_data, axis=0)
                label = np.append(label, w_length)
        return dataset, label
        
    def get_height(self):
        '''
        通过对垂直加速度积分并乘以窗口宽度获取deltH
        '''
        h = np.zeros((0))
        self.read_data()
        self.filter_data()
        steps = self.step_counter()
        w = 14 # 窗口宽度
        delt_t = (w+1) * 1 / self.freq # 一个窗口的时间 
        t = np.arange(0, delt_t, 1/self.freq) # 积分区间
        for i, v in enumerate(steps):
            if v['index'] > 7 and v['index'] + int(w/2) + 1 < self.filt_dataset.shape[0]:
                w_start = v['index'] - int(w/2) # 窗口起始索引
                w_end = v['index'] + int(w/2) + 1 # 窗口结束索引
                
                w_a_vertical = self.a_vertical[w_start: w_end]
                delt_v = integrate.simps(w_a_vertical, t)
                delt_h = np.abs(delt_v * delt_t)
                h = np.append(h, delt_h)
        return h
        


        
        
        
if __name__ == "__main__":
    '''
    测试代码
    '''
    dataPath_down = './data/SLEdata/down'
    dataPath_up = './data/SLEdata/up'
    freq = 25
    LD_down = LoadData(dataPath_down, freq)
    dataset_down, label_down = LD_down.build_dataset()
    
    LD_up = LoadData(dataPath_up, freq)
    dataset_up, label_up = LD_up.build_dataset()
    
    dataset_all, label_all = np.concatenate((dataset_down,  dataset_up), axis=0), \
                            np.concatenate((label_down, label_up), axis=0)
    print(dataset_all.shape)
    
    pre_height = LD_up.get_height()
    real_height = np.array([0.15]*pre_height.shape[0])
    height_mse = np.mean(np.abs(pre_height- real_height))
    print("预测高度平均误差:", height_mse)
    print(real_height.shape)
    
    # 滤波调参
    def adjust_filt():
        config = {
        "font.family":'Times New Roman',  # 设置字体类型
        #     "mathtext.fontset":'stix',
        }
        raw_acc = LD_down.raw_dataset[0:500, 0:3]
        raw_gyro = LD_down.raw_dataset[0:500, 3:6]
        filt_acc = LD_down.filter_dataset[0:500, 0:3]
        filt_gyro = LD_down.filter_dataset[0:500, 3:6]
        rcParams.update(config)
        plt.figure()
        # 时域图，未平滑
        plt.subplot(2, 2, 1)
        ax=plt.gca()
        ax.set_xlabel('T', fontsize=20)#设置横纵坐标标签
        ax.set_ylabel('Values', fontsize=20)
        x = np.arange(1, 501) # x轴坐标

        plt.plot(x, raw_acc[:, 0], label='x axix')
        plt.plot(x, raw_acc[:, 1], label='y axis')
        plt.plot(x, raw_acc[:, 2], label='z axis')
        plt.xticks(fontsize=18) #设置坐标轴刻度大小
        plt.yticks(fontsize=18)
        plt.legend()
        plt.title("raw_acc")
        
        # 时域图，平滑
        plt.subplot(2, 2, 3)
        ax=plt.gca()
        ax.set_xlabel('T', fontsize=20)#设置横纵坐标标签
        ax.set_ylabel('Values', fontsize=20)
        plt.plot(x, filt_acc[:, 0], label='x axix')
        plt.plot(x, filt_acc[:, 1], label='y axis')
        plt.plot(x, filt_acc[:, 2], label='z axis')
        plt.xticks(fontsize=18) #设置坐标轴刻度大小
        plt.yticks(fontsize=18)
        plt.legend()
        plt.title("filter_acc")
        
        plt.subplot(2, 2, 2)
        ax=plt.gca()
        ax.set_xlabel('T', fontsize=20)#设置横纵坐标标签
        ax.set_ylabel('Values', fontsize=20)
        plt.plot(x, raw_gyro[:, 0], label='x axix')
        plt.plot(x, raw_gyro[:, 1], label='y axis')
        plt.plot(x, raw_gyro[:, 2], label='z axis')
        plt.xticks(fontsize=18) #设置坐标轴刻度大小
        plt.yticks(fontsize=18)
        plt.legend()
        plt.title("raw_gyro")    
        
        plt.subplot(2, 2, 4)
        ax=plt.gca()
        ax.set_xlabel('T', fontsize=20)#设置横纵坐标标签
        ax.set_ylabel('Values', fontsize=20)
        plt.plot(x, filt_gyro[:, 0], label='x axix')
        plt.plot(x, filt_gyro[:, 1], label='y axis')
        plt.plot(x, filt_gyro[:, 2], label='z axis')
        plt.xticks(fontsize=18) #设置坐标轴刻度大小
        plt.yticks(fontsize=18)
        plt.legend()
        plt.title("filter_gyro")
        
        plt.show()
    # adjust_filt()

    #步数检测调参
    def adjust_steps_detection(a_vertical, steps):
        index_test = []
        value_test = []
        index_valley = []
        value_valley = []
        for v in steps:
            index_test.append(v['index'])
            value_test.append(v['acceleration'])
            index_valley.append(v['v_index'])
            value_valley.append(v['v_acceleration'])
        
        config = {
            "font.family":'Times New Roman',  # 设置字体类型
        #     "mathtext.fontset":'stix',
        }
        rcParams.update(config)
        textstr = '='.join(('steps', str(len(steps))))
        _, ax = plt.subplots(figsize=(15,8))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=20,
            verticalalignment='top', bbox=props)
        plt.plot(a_vertical)
        plt.scatter(index_test, value_test, color='r')
        plt.scatter(index_valley, value_valley, color='k', marker='^')
        plt.xlabel('samples', fontsize=20)
        plt.ylabel('Vertical Acceleration', fontsize=20)
        plt.xticks(fontsize=18) #设置坐标轴刻度大小
        plt.yticks(fontsize=18)
        plt.show()
        #plt.savefig('D:/硕士论文/图表/PDR竖直加速度.jpg',format='jpg',bbox_inches = 'tight',dpi=300)

    '''
    LD_down.read_data()
    LD_down.filter_data()
    a_vertical = LD_down.coordinate_conversion()
    steps = LD_down.step_counter()
    print(len(steps))
    adjust_steps_detection(a_vertical[0:390], steps[0:20])
    '''