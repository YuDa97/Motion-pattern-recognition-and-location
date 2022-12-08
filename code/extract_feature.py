import numpy as np

class GetFeature():
    def __init__(self, res_acc, win_acc):
        self.res_acc = res_acc # 合加速度
        self.win_acc = win_acc # 三轴加速度
        ## 合加速度时域特征
        self.mean_value = np.mean(self.res_acc)
        self.std_value = np.std(self.res_acc, ddof=1)
        self.max_value = np.max(self.res_acc)
        self.min_value = np.min(self.res_acc)
        self.var_value = np.var(self.res_acc, ddof=1)
        self.BeyondMean_value = self.CountBeyondMean()
        self.skew_value = self.CalSkew() # 偏度
        self.Kurt_value = self.CalKurt() # 峰度 

    def CountBeyondMean(self):
        '''
        计算窗口内合加速度超过均值次数
        input: 合加速度 ndarry 1-Dim
        output: 次数 int
        '''
        result = 0
        meanValue = np.mean(self.res_acc)
        for value in self.res_acc:
            if value > meanValue:
                result += 1
        return result

    def CalSkew(self):
        '''
        计算窗口内合加速度偏度
        input: 合加速度 ndarry 1-Dim
        output: skew float
        '''
        n = self.res_acc.shape[0]
        s = np.nan
        if n > 2: 
            ave = np.mean(self.res_acc)
            std = np.std(self.res_acc, ddof=1)
            temp_sum = sum(((self.res_acc - ave) / std) ** 3)
            s = temp_sum*n / ((n-1)*(n-2))
        return s
        
    def CalKurt(self):
        '''
        计算窗口内合加速度峰度
        input: 合加速度 ndarry 1-Dim
        output: Kurt float
        '''
        n = self.res_acc.shape[0]
        k = np.nan
        if n > 3:
            ave = np.mean(self.res_acc)
            std = np.std(self.res_acc, ddof=1)
            temp_sum = sum(((self.res_acc - ave) / std) ** 4)
            k = temp_sum*n*(n+1) / ((n-1)*(n-2)*(n-3)) - \
                3*(n-1)**2 / ((n-2)*(n-3))
        return k
