from zlib import Z_HUFFMAN_ONLY
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.ticker as mtick
from matplotlib import rcParams
import re
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

def plot_cdf(df):
    plt.figure(figsize=(20,10), dpi=100)
    plt.xticks(fontsize=25) #设置坐标轴刻度大小
    plt.yticks(fontsize=25)

    plt.grid(linestyle='-.') 
    ax=plt.gca()

    # 设置坐标轴的范围
    plt.xlim(0, 0.3)
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
    plt.show()
    #plt.savefig('./Figure/fusion_NSL_Continuous_Compare_cdf.jpg',format='jpg',bbox_inches = 'tight',dpi=300)

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


# plot_error_curve(KernelNumError)
plot_cdf(CNN_NSL_Continuous_Compare)
# plot_height_curve(Height_Compare)
