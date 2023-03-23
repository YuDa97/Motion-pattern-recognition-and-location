import numpy as np

class GetFeature():
    def __init__(self, res_acc, win_acc):
        self.res_acc = res_acc # 合加速度
        self.win_acc = win_acc # 三轴加速度
        ## 合加速度时域特征, 共10个
        self.mean_value = np.mean(self.res_acc)
        self.std_value = np.std(self.res_acc, ddof=1)
        self.max_value = np.max(self.res_acc)
        self.min_value = np.min(self.res_acc)
        self.var_value = np.var(self.res_acc, ddof=1)
        self.median_value = np.median(self.res_acc)
        self.BeyondMean_value = self.CountBeyondMean()
        self.skew_value = self.CalSkew(self.res_acc) # 偏度
        self.Kurt_value = self.CalKurt(self.res_acc) # 峰度 
        self.iqr = self.CalIqr(self.res_acc) # 四分位距
        ## 合加速度频域特征, 4个
        self.first_max_amplitude, self.first_max_amplitude_fre, \
        self.second_max_amplitude, self.second_max_amplitude_fre = self.FreqAnalysis(self.res_acc) 
        ## 三轴加速度时域特征, 共30个
        ### 均值
        self.mean_x = np.mean(self.win_acc[:, 0])
        self.mean_y = np.mean(self.win_acc[:, 1]) 
        self.mean_z = np.mean(self.win_acc[:, 2])
        ### 标准差
        self.std_x = np.std(self.win_acc[:, 0])
        self.std_y = np.std(self.win_acc[:, 1]) 
        self.std_z = np.std(self.win_acc[:, 2])
        ### 最大值
        self.max_x = np.max(self.win_acc[:, 0]) 
        self.max_y = np.max(self.win_acc[:, 1]) 
        self.max_z = np.max(self.win_acc[:, 2])
        ### 最小值
        self.min_x = np.min(self.win_acc[:, 0])
        self.min_y = np.min(self.win_acc[:, 1]) 
        self.min_z = np.min(self.win_acc[:, 2])
        ### 方差
        self.var_x = np.var(self.win_acc[:, 0])
        self.var_y = np.var(self.win_acc[:, 1])
        self.var_z = np.var(self.win_acc[:, 2])
        ### 中位数
        self.median_x = np.median(self.win_acc[:, 0])
        self.median_y = np.median(self.win_acc[:, 1])
        self.median_z = np.median(self.win_acc[:, 2])
        ### 偏度
        self.skew_x = self.CalSkew(self.win_acc[:, 0])
        self.skew_y = self.CalSkew(self.win_acc[:, 1])
        self.skew_z = self.CalSkew(self.win_acc[:, 2])
        ### 峰度
        self.Kurt_x = self.CalKurt(self.win_acc[:, 0])
        self.Kurt_y = self.CalKurt(self.win_acc[:, 1])
        self.Kurt_z = self.CalKurt(self.win_acc[:, 2])
        ### 四分位间距
        self.iqr_x = self.CalIqr(self.win_acc[:, 0])
        self.iqr_y = self.CalIqr(self.win_acc[:, 1])
        self.iqr_z = self.CalIqr(self.win_acc[:, 2])
        ### 互相关系数(Pearson相关系数)
        self.p_xy = self.CalPearson(self.win_acc, 'xy')
        self.p_xz = self.CalPearson(self.win_acc, 'xz')
        self.p_yz = self.CalPearson(self.win_acc, 'yz')
        ## 合加速度频域特征
        
        
        
        
    def CountBeyondMean(self):
        '''
        计算窗口内数据超过均值次数
        input:  ndarry 1-Dim
        output: 次数 int
        '''
        result = 0
        meanValue = np.mean(self.res_acc)
        for value in self.res_acc:
            if value > meanValue:
                result += 1
        return result

    def CalSkew(self, data):
        '''
        计算窗口内数据偏度
        input:  ndarry 1-Dim
        output: skew float
        '''
        n = data.shape[0]
        s = np.nan
        if n > 2: 
            ave = np.mean(data)
            std = np.std(data, ddof=1)
            temp_sum = sum(((data - ave) / std) ** 3)
            s = temp_sum*n / ((n-1)*(n-2))
        return s
        
    def CalKurt(self, data):
        '''
        计算窗口内合加速度峰度
        input: data ndarry 1-Dim
        output: Kurt float
        '''
        n = data.shape[0]
        k = np.nan
        if n > 3:
            ave = np.mean(data)
            std = np.std(data, ddof=1)
            temp_sum = sum(((data - ave) / std) ** 4)
            k = temp_sum*n*(n+1) / ((n-1)*(n-2)*(n-3)) - \
                3*(n-1)**2 / ((n-2)*(n-3))
        return k
    
    def CalIqr(self, data): 
        '''
        计算四分位距
        '''
        
        qr1 = np.quantile(data, 0.25)  # 下四分位数
        qr3 = np.quantile(data, 0.75)  # 上四分位数
        iqr = qr3 - qr1  # 计算四分位距
        return iqr

    def CalPearson(self, data, axis):
        '''
        计算各轴加速度间相关性系数
        input: data ndarry 2-Dim
        '''
        data = data.T
        coorr_matrix = np.corrcoef(data)
        if axis == 'xy' or 'yx':
            return coorr_matrix[0, 1]
        elif axis == 'xz' or 'zx':
            return coorr_matrix[0, 2]
        elif axis == 'yz' or 'zy':
            return coorr_matrix[1, 2]
        raise Exception("axis只能是x,y,z除了自身以外的组合")
    
    def FreqAnalysis(self, data):
        '''
        频域分析函数
        input: data ndarry 1-D, 合加速度数据

        '''
        
        n = data.shape[0]
        if n > 3:
            fourier = np.fft.rfft(data)
            sample_rate = 50
            abs_data = np.abs(fourier).tolist()
            freq = np.fft.rfftfreq(n, d=1./sample_rate)
            sort_fourier = np.sort(abs_data[1:])
            
            first_max_abs = sort_fourier[-1]
            first_max_freq = freq[abs_data.index(first_max_abs)]
            
            second_max_abs = sort_fourier[-2]
            second_max_freq = freq[abs_data.index(second_max_abs)]
            
            return first_max_abs, first_max_freq, second_max_abs, second_max_freq
        else:
            return np.nan, np.nan, np.nan, np.nan
            

        