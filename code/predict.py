# 建立训练集
import build_dataset as bd
# 数据处理
import numpy as np
import pandas as pd
# 特征选择
from feature_selector import FeatureSelector
# 分类器
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


## 数据准备
path = 'D:/motion sense/Motion-pattern-recognition/data/TrainData'
freq = 25 # 数据采样频率是25Hz
label_coding = {'stand': 0, 'walk': 1, 'up': 2, 'down': 3}
feature_num = 8 
training_dimention = feature_num + 1
startidx = 70 # 舍掉前70个点
window_wide = int(1.5 * freq) # 滑动窗口宽度

training_set, all_feature_name = bd.creat_training_set(path, label_coding, startidx, window_wide, training_dimention)
# testing_set, testing_feature_name = bd.creat_testing_set(path)
train_x, train_y = training_set[:, 0:feature_num], training_set[:, -1]
# test_x, true_y = testing_set[:, 0:feature_num], testing_set[:, -1]

## 特征选择
df_train_x = pd.DataFrame(data=train_x, columns=all_feature_name) # 将训练集转化为datafram格式,作为feature_selector输入

fs = FeatureSelector(data = df_train_x, labels = train_y) 
fs.identify_collinear(correlation_threshold=0.0000005, one_hot=False)
fs.identify_zero_importance(task = 'classification', n_iterations = 10, early_stopping = False)
fs.identify_low_importance(cumulative_importance=0.99)
selected_training_set = fs.remove(methods = ['collinear', 'zero_importance', 'low_importance']) 
remain_features = list(selected_training_set) # 查看保留的特征
# removed_features  = fs.check_removal() # 查看移除的特征
# print(removed_features)
train_x = selected_training_set[remain_features].values # 筛选特征后的训练集


#df_test_x = pd.DataFrame(data=train_x, columns=all_feature_name) # 将测试集转化为datafram格式
#test_x = df_test_y[remain_features].values # 筛选特征后的测试集

## SVM分类
svc = SVC(C=100, kernel="rbf", max_iter=1000)
svc.fit(train_x,train_y)

#predictions_svc = svc.predict(test_x)
#print('SVC_Accuracy: {}'.format((true_y == predictions_svc).mean()))

## RandomForest分类
R_tree = RandomForestClassifier(n_estimators=100)
R_tree.fit(train_x,train_y)
#predictions_R_tree = R_tree.predict(test_x)
#print('RandomForest_Accuracy: {}'.format((true_y == predictions_R_tree).mean()))

## K-NN分类
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(train_x,train_y)
#predictions_knn = knn.predict(test_x)
#print('KNN_Accuracy: {}'.format((true_y == predictions_knn).mean()),"\n")