# 建立训练集
import recognition.build_dataset as bd
# 数据处理
import numpy as np
import pandas as pd
# 特征选择
from recognition.feature_selector import FeatureSelector
# 分类器
from sklearn.svm import SVC
# pdr定位
import location.pdr as pdr
# 数据平滑
from recognition.build_dataset import smooth_data

def change_prediction(predictions, anomalies_index):
    '''
    对分类器预测结果进行修正
    input: 
    predictions->ndarry: 分类器预测结果
    anomalies_index->list: 发生突变窗口的下标
    output:
    correct_predictions->ndarry: 修正后的预测结果
    '''
    for index in anomalies_index:
        pass

# 运动模式识别部分
## 数据准备
train_path = 'D:/motion sense/Motion-pattern-recognition/data/TrainData'
test_path = 'D:/motion sense/Motion-pattern-recognition/data/TestData/exp1'
freq = 25 # 数据采样频率是25Hz
label_coding = {'stand': 0, 'walk': 1, 'up': 2, 'down': 3}
feature_num = 44
training_dimention = feature_num + 1
startidx = 75 # 舍掉前75个点
window_wide = int(1.5 * freq) # 滑动窗口宽度

### 对测试集和加速度小波变换后进行频域分析
test_res_acc = bd.creat_anomalies_detect_dataset(test_path)
cwtmatr ,frequencies = bd.cwt_data(test_res_acc)
anomalies_index = bd.anomalies_detect(abs(cwtmatr[8]), window_wide,showFigure=False) # 发生突变的窗口下标


train_set, all_feature_name = bd.creat_training_set(train_path, label_coding, startidx, window_wide, training_dimention)
test_set = bd.creat_testing_set(test_path, label_coding, startidx, freq, window_wide, training_dimention)
train_x, train_y = train_set[:, 0:feature_num], train_set[:, -1]
test_x, true_y = test_set[:, 0:feature_num], test_set[:, -1]

## 特征选择
df_train_x = pd.DataFrame(data=train_x, columns=all_feature_name) # 将训练集转化为datafram格式,作为feature_selector输入

fs = FeatureSelector(data = df_train_x, labels = train_y) 
fs.identify_collinear(correlation_threshold=0.0000005, one_hot=False)
fs.identify_zero_importance(task = 'classification', n_iterations = 10, early_stopping = False)
fs.identify_low_importance(cumulative_importance=0.99)
selected_training_set = fs.remove(methods = ['zero_importance', 'low_importance']) # 可选'collinear', 'zero_importance', 'low_importance'
remain_features = list(selected_training_set) # 查看保留的特征
# removed_features  = fs.check_removal() # 查看移除的特征
# print(removed_features)
train_x = selected_training_set[remain_features].values # 筛选特征后的训练集


df_test_x = pd.DataFrame(data=test_x, columns=all_feature_name) # 将测试集转化为datafram格式
test_x = df_test_x[remain_features].values # 筛选特征后的测试集

## SVM分类
svc = SVC(C=100, kernel="rbf", max_iter=1000)
svc.fit(train_x,train_y)
 
predictions_svc = svc.predict(test_x)
print('SVC_Accuracy: {}'.format((true_y == predictions_svc).mean()))

# pdr定位
## 数据准备
walking_data_file = test_path + '/pdr_data.csv'
df_walking = pd.read_csv(walking_data_file)

linear = df_walking[[col for col in df_walking.columns if 'linear' in col]].values[startidx:]
gravity = df_walking[[col for col in df_walking.columns if 'gravity' in col]].values[startidx:]
rotation = df_walking[[col for col in df_walking.columns if 'rotation' in col]].values[startidx:]
## 数据平滑
linear = smooth_data(linear)
gravity = smooth_data(gravity)
rotation = smooth_data(rotation)

pdr = pdr.Model(linear, gravity, rotation)
pdr.show_trace(frequency=25, walkType='normal', initPosition=(0, 0, 0),\
                predictPattern=predictions_svc, m_WindowWide=window_wide)
