# Motion-pattern-recognition
本项目实现运动模式识别,人迹航位(PDR)定位和WiFi指纹定位以及EKF融合定位（均是三维情况）    
运动模式识别过程包括:数据采集->滤波->特征提取/筛选->构建训练/测试集->训练分类器->预测  
PDR定位过程包括:数据采集->滤波->步数检测->步长推断,航向角推算->输出定位结果  
WiFi定位过程包括:数据采集->滤波->构建指纹库->选择/训练匹配算法->输出定位结果
## 文件说明
1. **code部分**  
+ CNNParameter:保存训练好的CNN模型参数(使用GPU训练，需要gpu版本pytorch加载)。  
+ RemainFeature:保存lgb筛选出的加速度特征。  
+ recognition:运动模式识别模块
   + build_dataset.py:准备数据
   + extract_feature.py:提取加速度计数据各种统计特征用于分类
   + feature_selector.py:包含特征选择算法(皮尔逊相关性系数，lgb)
   + select_feature.py:根据需求选择出特征
+ SLE:Step Length Estimation上下楼步长推断模块  
   + CNN.py:定义CNN网络模型结构  
   + preparedata.py:准备用于训练CNN的数据,并通过对垂直方向加速度积分计算高度变化量
   + predict.py:推测上下楼步长  
+ location:wifi指纹定位和PDR定位模块  
   +  fusion.py:PDR与wifi融合模块，采用的是EKF算法融合
   +  pdr.py:PDR定位模块,包含步数检测、步长估计、航向角解算算法
   +  ukf.py:ukf融合模块
   +  wifi.py:包含wifi指纹定位算法
+ demo_xxxx.py:对每一模块进行独立调试，可用于寻找合适的超参数
+ motion_predict.py:实现整个运动模式识别流程
+ main.py:实现运动模式识别与pdr三维定位
2. **data部分**  
+ SLEdata:用于训练/测试上下楼步长估计模型
+ TestData:用于运动模式识别和PDR流程测试
+ TrainData:用于运动模式识别分类器的训练
+ demo:为项目早期调试代码数据  
+ FuseLocationTestData:用于运动模式识别、PDR、WiFi、和EKF融合定位流程测试  
+ WiFi:保存RP的RSSI和TP的RSSI
## 注意事项  
本项目所有数据均由智能手机收集，要求收集时始终保持手持手机于胸前，手机y轴朝向正前方，z轴与重力方向相反。采样频率为25Hz。![手机坐标系与地球坐标系](https://github.com/YuDa97/Motion-pattern-recognition/blob/main/Figure/phone&ENU_coordinate.jpg)
1. **运动模式识别部分**
只选用三轴加速度计数据进行训练和预测，特征提取也仅限于常见统计特征，特征筛选主要依靠lgb贡献率筛选，主要的分类器为SVM。
2. **PDR定位部分**  
建立本地坐标系时，需要将y轴与正北方向对齐，z轴与重力反方向对齐，若不对齐解算出的航向角有误，预测位置需要进行坐标转换  
在本项目建立的坐标系中，一个单位对应现实中的0.6m
3. **wifi定位部分**  
该部分测试点RSS的收集过程并非在行人行走过程中收集，需要在每一个测试点静止站立1分钟来获取稳定的RSS
## 结果
对./data/TestData/exp1数据进行测试  
SVM模式识别precision为0.94  
pdr一共检测出步数132步，平均定位误差为1.26m（欧式距离）
![pdr定位轨迹](https://github.com/YuDa97/Motion-pattern-recognition/blob/main/Figure/pdr_trace.png)
## 鸣谢
感谢salmoshu大佬的[Location项目](https://github.com/salmoshu/location)，对我们帮助很大！
