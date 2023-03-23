# 建立训练集
import recognition.build_dataset as bd
from sklearn.model_selection import train_test_split
# 数据处理
import numpy as np
import pandas as pd
# 特征选择
from recognition.feature_selector import FeatureSelector
# 分类器
import lightgbm as lgb
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
# 分类报告
from sklearn import metrics 
# 运行时间计算
import time as t


## 数据准备 
train_path = '/home/yuda/Motion-pattern-recognition/data/TrainData'
test_path = '/home/yuda/Motion-pattern-recognition/data/TestData/exp1'
freq = 25 # 数据采样频率是25Hz
label_coding = {'stand': 0, 'walk': 1, 'up': 2, 'down': 3}
feature_num = 44
training_dimention = feature_num + 1
startidx = 75 # 舍掉前75个点
window_wide = int(1.5 * freq) # 滑动窗口宽度
split_trainingset = False # 是否将训练数据划分为测试集和训练集,不对连续运动状态进行识别

'''
#效果不好，未启用
### 对测试集和加速度小波变换后进行频域分析
test_res_acc = bd.creat_anomalies_detect_dataset(test_path)
cwtmatr ,frequencies = bd.cwt_data(test_res_acc)
anomalies_index = bd.anomalies_detect(abs(cwtmatr[8]), window_wide,showFigure=False) # 发生突变的窗口下标
'''

train_set, all_feature_name = bd.creat_training_set(train_path, label_coding, startidx, window_wide, training_dimention)
if split_trainingset:
    dataset, label = train_set[:, 0:feature_num], train_set[:, -1]
    train_x, test_x, train_y, true_y  = train_test_split(
        dataset, label, test_size=0.2, random_state=42)

else:
    test_set = bd.creat_testing_set(test_path, label_coding, startidx, freq, window_wide, training_dimention)
    train_x, train_y = train_set[:, 0:feature_num], train_set[:, -1]
    test_x, true_y = test_set[:, 0:feature_num], test_set[:, -1]


## 特征选择

df_train_x = pd.DataFrame(data=train_x, columns=all_feature_name) # 将训练集转化为datafram格式,作为feature_selector输入

fs = FeatureSelector(data = df_train_x, labels = train_y)

fs.identify_collinear(correlation_threshold=0.8, one_hot=False)
fs.identify_zero_importance(task = 'classification', n_iterations = 10, early_stopping = False)
fs.identify_low_importance(cumulative_importance=0.99)
#selected_training_set = fs.remove(methods = ['collinear'])
selected_training_set = fs.remove(methods = ['zero_importance']) # 可选'collinear', 'zero_importance', 'low_importance'
remain_features = list(selected_training_set) # 查看保留的特征
# removed_features  = fs.check_removal() # 查看移除的特征
# print(removed_features)
fs.plot_feature_importances(plot_n=40, threshold=0.9) # 画出特征重要性排序
train_x = selected_training_set[remain_features].values # 筛选特征后的训练集


df_test_x = pd.DataFrame(data=test_x, columns=all_feature_name) # 将测试集转化为datafram格式
test_x = df_test_x[remain_features].values # 筛选特征后的测试集

## lgb分类
lgbModel = lgb.LGBMClassifier(n_estimators=1000, learning_rate = 0.05, verbose = -1)
lgbModel.fit(train_x,train_y)

lgb_start_time = t.time()
predictions_lgb = lgbModel.predict(test_x)
lgb_end_time = t.time()
print('lgb分类报告: \n', metrics.classification_report(true_y, predictions_lgb))
print("lgb_predict_time", lgb_end_time - lgb_start_time)

## SVM分类
svc = SVC(C=300, kernel="rbf", max_iter=1000)
svc.fit(train_x,train_y)

svc_start_time = t.time()
predictions_svc = svc.predict(test_x)
svc_end_time = t.time()
print('SVM分类报告: \n', metrics.classification_report(true_y, predictions_svc ))
print("svc_predict_time", svc_end_time - svc_start_time)

## RandomForest分类
R_tree = RandomForestClassifier(n_estimators=100)
R_tree.fit(train_x,train_y)

rTree_start_time = t.time()
predictions_R_tree = R_tree.predict(test_x)
rTree_end_time = t.time()
print('RandomForest分类报告: \n', metrics.classification_report(true_y, predictions_R_tree))
print("rTree_predict_time", rTree_end_time - rTree_start_time)

## K-NN分类
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(train_x,train_y)
knn_start_time = t.time()
predictions_knn = knn.predict(test_x)
knn_end_time = t.time()
print('K-NN分类报告: \n', metrics.classification_report(true_y, predictions_knn))
print("K-NN_predict_time", knn_end_time - knn_start_time)
