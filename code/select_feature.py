import build_dataset as bd
import numpy as np
import pandas as pd
from feature_selector import FeatureSelector

path = 'D:/motion sense/Motion-pattern-recognition/data/TrainData'
freq = 25 # 数据采样频率是25Hz
label_coding = {'stand': 0, 'walk': 1, 'up': 2, 'down': 3}
feature_num = 44
training_dimention = feature_num + 1
startidx = 70 # 舍掉前70个点
window_wide = int(1.5 * freq) # 滑动窗口宽度

training_set, training_feature_name = bd.creat_training_set(path, label_coding, startidx, window_wide, training_dimention)
train_x, train_y = training_set[:, 0:feature_num], training_set[:, -1]
df_train_x = pd.DataFrame(data=train_x, columns=training_feature_name) # 将训练集转化为datafram格式,作为feature_selector输入
fs = FeatureSelector(data = df_train_x, labels = train_y)

# 选择相关性大于指定值(通过correlation_threshold指定值)的feature
fs.identify_collinear(correlation_threshold=0.0000005, one_hot=False) 
correlated_features = fs.ops['collinear']
print(correlated_features[:5])

# 选择zero importance的feature,即对模型预测结果无贡献的特征
#
# 参数说明：
#          task: 'classification' / 'regression', 如果数据的模型是分类模型选择'classificaiton',
#                否则选择'regression'
#          eval_metric: 判断提前停止的metric. for example, 'auc' for classification, and 'l2' for regression problem
#          n_iteration: 训练的次数
#          early_stopping: True/False, 是否需要提前停止
fs.identify_zero_importance(task = 'classification', 
                            n_iterations = 10, early_stopping = False)
zero_importance_features = fs.ops['zero_importance']
print(zero_importance_features) # 查看选择出的zero importance feature

# 对模型预测结果只有很小贡献的特征
# 选择出对importance累积和达到99%没有贡献的feature
fs.identify_low_importance(cumulative_importance=0.99)

# 查看选择出的feature
fs.ops['low_importance']