from zlib import Z_HUFFMAN_ONLY
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.ticker as mtick
from matplotlib import rcParams
import re
from mpl_toolkits.mplot3d import Axes3D
#%matplotlib inline
config = {
    "font.family":'Times New Roman',  # 设置字体类型
#     "mathtext.fontset":'stix',
}
rcParams.update(config)

KernelNumError = pd.read_excel('./runs/CNNParameterTuning/DifferentKernelNums.xlsx')
KernelSizeError = pd.read_excel('./runs/CNNParameterTuning/DifferentKernelSize.xlsx')
CNN_NSL_Disperse_Compare = pd.read_excel('./runs/DisperseStepLengthResultCompare.xlsx')
CNN_NSL_Continuous_Compare = pd.read_excel('./runs/ContinuousStepLengthResultCompare.xlsx')
Height_Compare = pd.read_excel('./runs/HeightCompare.xlsx')
Location_Results_Compare = pd.read_excel('./runs/LocationResultCompare.xlsx')
Realtrace = pd.read_csv('./data/TestData/test_coordinate.csv')

fusion_step_location_predict = Location_Results_Compare.loc[:,['x_fusion', 'y_fusion','z_fusion']]
NSL_step_location_predict = Location_Results_Compare.loc[:,['x_NSL', 'y_NSL', 'z_NSL']]
location_result_cdf_compare = Location_Results_Compare.loc[:, ['error_fusion', 'error_NSL']]
def plot_cdf(df):
    plt.figure(figsize=(20,10), dpi=100)
    plt.xticks(fontsize=25) #设置坐标轴刻度大小
    plt.yticks(fontsize=25)

    plt.grid(linestyle='-.') 
    ax=plt.gca()

    # 设置坐标轴的范围
    #plt.xlim(0, 0.3)
    #plt.ylim(0, 1)
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))#设置横坐标刻度保留2位小数
    ax.spines['bottom'].set_linewidth(1.2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(1.2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(1.2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(1.2)
    ax.set_xlabel('MSE(m)', fontsize=25)#设置横纵坐标标签
    ax.set_ylabel('Cumulative Probability Distribution ', fontsize=25)

    for col in df.columns:
        if df[col].isnull().any() == False: 
            sns.ecdfplot(df[col],label=col,linewidth=2)

    plt.legend(fontsize = 20,bbox_to_anchor=(0.85,0.8)) #显示图例，字体为20
    #plt.show()
    plt.savefig('./Figure/fusion_NSL_Continuous_Location_Compare_cdf.jpg',format='jpg',bbox_inches = 'tight',dpi=300)

def plot_error_curve(df):
    plt.figure(figsize=(20,10), dpi=100)
    plt.xticks(fontsize=25) #设置坐标轴刻度大小
    plt.yticks(fontsize=25)

    plt.grid(linestyle='-.') 
    ax=plt.gca()

    # 设置坐标轴的范围
    #plt.xlim(8, 12)
    #plt.ylim(0, 0.1)
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))#设置横坐标刻度保留2位小数
    ax.spines['bottom'].set_linewidth(1.2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(1.2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(1.2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(1.2)
    ax.set_xlabel('Kernel Nums', fontsize=25)#设置横纵坐标标签
    ax.set_ylabel('MSE(m) ', fontsize=25)

    x, y = np.array([]), np.array([])
    for col in df.columns:
        if df[col].isnull().any() == False: 
           x = np.append(x, int(re.findall(r'\d+', col)[0]))
           y = np.append(y, df[col].mean())

    plt.plot(x, y, 'o-')
    #plt.legend(fontsize = 20,bbox_to_anchor=(0.85,0.8)) #显示图例，字体为20
    plt.show()
    #plt.savefig(f'./Figure/KernelSizeError_curve.jpg',format='jpg',bbox_inches = 'tight',dpi=300)

def plot_height_curve(df):
    plt.figure(figsize=(20,10), dpi=100)
    plt.xticks(fontsize=25) #设置坐标轴刻度大小
    plt.yticks(fontsize=25)

    plt.grid(linestyle='-.') 
    ax=plt.gca()
    # 设置坐标轴的范围
    #plt.xlim(8, 12)
    #plt.ylim(0, 0.1)
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))#设置横坐标刻度保留2位小数
    ax.spines['bottom'].set_linewidth(1.2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(1.2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(1.2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(1.2)
    ax.set_xlabel('Step', fontsize=25)#设置横纵坐标标签
    ax.set_ylabel('Height(m)', fontsize=25)

    for col in df.columns:
        if df[col].isnull().any() == False: 
           y = df[col]
           x = np.arange(y.shape[0])
           plt.plot(x, y, 'o-', label=col)
    plt.legend(fontsize = 20,bbox_to_anchor=(0.85,0.8)) #显示图例，字体为20
    #plt.show()
    plt.savefig(f'./Figure/Height_compare.jpg',format='jpg',bbox_inches = 'tight',dpi=300)

def plot_location_trace(real_trace, fuse_location, NSL_location):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        plt.grid()
        handles = []
        labels = []

        
        trace_x = real_trace.loc[:,'x']
        trace_y = real_trace.loc[:,'y']
        trace_z = real_trace.loc[:,'z']
        l1, = ax.plot(trace_x, trace_y, trace_z, color='orange')
        handles.append(l1)
        labels.append('Real Trace')
       
        fuse_x = fuse_location.loc[:,'x_fusion']
        fuse_y = fuse_location.loc[:,'y_fusion']
        fuse_z = fuse_location.loc[:,'z_fusion']
        l2, = ax.plot(fuse_x, fuse_y, fuse_z, 'o-')
        handles.append(l2)
        labels.append('FuseModel-Based PDR Location')

        NSL_x = NSL_location.loc[:,'x_NSL']
        NSL_y = NSL_location.loc[:,'y_NSL']
        NSL_z = NSL_location.loc[:,'z_NSL']
        l3, = ax.plot(NSL_x, NSL_y, NSL_z, '*-')
        handles.append(l3)
        labels.append('NSL-Based PDR Location')


        ax.set_xlabel('X', fontsize=18)#设置横纵坐标标签
        ax.set_ylabel('Y', fontsize=18)
        ax.set_zlabel('Z', fontsize=18)
        plt.legend(handles=handles,labels=labels,loc='best',fontsize = 20)
        ax.tick_params(labelsize=14) #设置坐标轴刻度大小
        plt.show()
        #plt.savefig('./Figure/pdr_location_trace.jpg',bbox_inches = 'tight')

# plot_error_curve(KernelNumError)
# plot_cdf(location_result_cdf_compare)
# plot_height_curve(Height_Compare)
plot_location_trace(Realtrace, fusion_step_location_predict, NSL_step_location_predict)