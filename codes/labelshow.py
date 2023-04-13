import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

random.seed(43)
plt.style.use(['science', 'grid', 'no-latex'])  # 启用科学风格
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.rc('font', family='Times New Roman')

# 载入训练集的label数据
def load_data():
    print('load data\n')  # 服务器进度展示

    # 载入训练集的y
    with open("/home/guozy/Ni-YSZ_CNN/train/Results_all.txt", "r") as f:
        line = f.readline()
        label_list = []
        while line:
            num = list(map(float, line.split()))
            label_list.append(num)
            line = f.readline()
        f.close()
        train_y = np.array(label_list)
        
    # 载入测试集的y
    with open("/home/guozy/Ni-YSZ_CNN/predict/Results_15000_predict.txt", "r") as f:
        line = f.readline()
        label_list = []
        while line:
            num = list(map(float, line.split()))
            label_list.append(num)
            line = f.readline()
        f.close()
        test_y = np.array(label_list)

    return train_y,test_y


# 展示数据集的label
def show_data(train_y, name):
    print('show data\n')  # 服务器进度展示
    plt.figure(dpi=500,figsize=(15,10))
    palette = plt.get_cmap('tab20c')  # 'Pastel2') # 'Set1'
    
    
    title = [r'$E$',r'${N}$',r'$G$',r'CTE',r'$L_{\rm{TPB}}$']
    x_label=[r'MPa',r'value',r'MPa',r'K$^{-1}$',r'$\rm{{\mu}m^{-2}}$']
    
    for i in range(5):
        ax=plt.subplot(3, 2, i+1)
        sns.histplot(train_y[:, i], kde=True, shrink=0.8, bins=20, color=palette.colors[4*i])
        plt.title(title[i],{'size' : 12})
        plt.xlabel(x_label[i],{'size' : 12})
        plt.ylabel('Count',{'size' : 12})
    
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,wspace=0.17, hspace=0.40)
    plt.suptitle(name, fontsize=22,x=0.51,y=0.93)
    plt.savefig('/home/guozy/Ni-YSZ_CNN/train/' + name + '.png')  # 保存训练过程的图片

# 展示归一化后数据集的label
def show_nol_data(train_y, name):
    print('show data\n')  # 服务器进度展示
    plt.figure(dpi=500,figsize=(15,10))
    palette = plt.get_cmap('tab20c')  # 'Pastel2') # 'Set1'
    
    
    title = [r'$E^*$',r'${N}^*$',r'$G^*$',r'CTE$^*$',r'$L_{\rm{TPB}}^*$']
    
    for i in range(5):
        ax=plt.subplot(3, 2, i+1)
        sns.histplot(train_y[:, i], kde=True, shrink=0.8, bins=20, color=palette.colors[4*i])
        plt.title(title[i],{'size' : 12})
        plt.ylabel('Count',{'size' : 12})
    
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,wspace=0.17, hspace=0.40)
    plt.suptitle(name, fontsize=22,x=0.51,y=0.93)
    plt.savefig('/home/guozy/Ni-YSZ_CNN/train/' + name + '.png')  # 保存训练过程的图片




# 数据预处理
def process(train_y):
    print('process data\n')  # 服务器进度展示

    # y归一化到[0,1]
    train_max = np.max(train_y, axis=0)
    train_min = np.min(train_y, axis=0)
    for i in range(train_y.shape[1]):
        train_y[:, i] = (train_y[:, i] - train_min[i]) / (train_max[i] - train_min[i])

    return train_y


if __name__ == '__main__':
    train_y, test_y= load_data()  # 载入训练集和数据集的数据
    #show_data(train_y, 'Distribution of all Train Labels')
    train_y = process(train_y)
    show_nol_data(train_y, 'Normalized Distribution of all Train Labels')
    #show_data(test_y, 'Distribution of all Test Labels')
