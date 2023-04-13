import numpy as np
import pickle
from keras.models import load_model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from PIL import Image
import os
import time
import random
import tensorflow as tf
from matplotlib import pyplot as plt
random.seed(43)
plt.style.use(['science','grid','no-latex'])    # 启用科学风格

# 数据增强中的图像旋转
def rotate(xi):
    axis = random.randint(1, 3)  # 选择固定的轴：1是x，2是y，3是z
    step = random.randint(1, 3)  # 选择逆时针转动的次数为1-3次
    x_aug = np.zeros([1, 64, 64, 64, 1])  # 增强后的样本x

    # 分9种情况进行讨论
    if axis == 1 and step == 1:
        for i in range(64):
            x_aug[0, i, :, :, 0] = np.rot90(xi[i, :, :], step)

    elif axis == 1 and step == 2:
        for i in range(64):
            x_aug[0, i, :, :, 0] = np.rot90(xi[i, :, :], step)

    elif axis == 1 and step == 3:
        for i in range(64):
            x_aug[0, i, :, :, 0] = np.rot90(xi[i, :, :], step)

    elif axis == 2 and step == 1:
        for i in range(64):
            x_aug[0, :, i, :, 0] = np.rot90(xi[:, i, :], step)

    elif axis == 2 and step == 2:
        for i in range(64):
            x_aug[0, :, i, :, 0] = np.rot90(xi[:, i, :], step)

    elif axis == 2 and step == 3:
        for i in range(64):
            x_aug[0, :, i, :, 0] = np.rot90(xi[:, i, :], step)

    elif axis == 3 and step == 1:
        for i in range(64):
            x_aug[0, :, :, i, 0] = np.rot90(xi[:, :, i], step)

    elif axis == 3 and step == 2:
        for i in range(64):
            x_aug[0, :, :, i, 0] = np.rot90(xi[:, :, i], step)

    elif axis == 3 and step == 3:
        for i in range(64):
            x_aug[0, :, :, i, 0] = np.rot90(xi[:, :, i], step)

    return x_aug


# 镜像
def mirror(xi):
    choice=random.randint(1, 6)  # 选择镜像的面
    x_aug = np.zeros([1, 64, 64, 64, 1])  # 增强后的样本x

    # 分9种情况进行讨论
    if choice == 1: # x轴正方向的镜像
        for i in range(64):
            x_aug[0, :, :, i, 0] =np.flip(xi[:, :, i],axis=0)

    elif choice == 2:
        for i in range(64):
            x_aug[0, :, :, i, 0] =np.flip(xi[:, :, i],axis=1)

    elif choice == 3:
        for i in range(64):
            x_aug[0, i, :, :, 0] =np.flip(xi[i, :, :],axis=0)

    elif choice == 4:
        for i in range(64):
            x_aug[0, i, :, :, 0] =np.flip(xi[i, :, :],axis=1)

    elif choice == 5:
        for i in range(64):
            x_aug[0, :, i, :, 0] =np.flip(xi[:, i, :],axis=0)

    elif choice == 6:
        for i in range(64):
            x_aug[0, :, i, :, 0] =np.flip(xi[:, i, :],axis=1)

    return x_aug


# 数据增强
def aug_data(x, y):
    print('augment data\n')
    total = x.shape[0]  # 总共有多少幅图

    for i in range(total):
        xi = x[i, :, :, :, 0]  # 得到第i个样本
        choice = random.randint(1, 15)  # 选择镜像的面
        if choice <= 9:
            x_aug = rotate(xi)  # 获得第i个样本对应的随机旋转后的数据
        else:
            x_aug = mirror(xi)
        # 将数据增强后的样本和原来的样本合并
        x = tf.concat([x, x_aug], axis=0)

    y = tf.concat([y, y], axis=0)

    return x.numpy(), y.numpy()


def load_data():
    print ('load  data')
    ## 载入测试集的x
    count = -1
    final_test_data = np.zeros([344, 64, 64, 64, 1])
    for cycleTiff in range(357):
        filename = "/home/guozy/Ni-YSZ_CNN/predict/anode_stacks_15000/" + str(cycleTiff + 1) + ".tiff"
        if os.path.exists(filename) == False:
            continue
        count = count + 1
        img = Image.open(filename)
        for i in range(img.n_frames):  # 可以用img.n_frames来得到总页数
            try:
                img.seek(i)  # 是否存在
                final_test_data[count, :, :, i, 0] = img
            except EOFError:  # 页数读完了
                break

    # 载入测试集的y
    with open("/home/guozy/Ni-YSZ_CNN/predict/Results_15000_predict.txt", "r") as f:
        line = f.readline()
        label_list = []
        while line:
            num = list(map(float, line.split()))
            label_list.append(num)
            line = f.readline()
        f.close()
        final_test_label = np.array(label_list)
        
    return final_test_data,final_test_label
    
# 载入选择的模型
def get_model():
    print('avaliable model:')
    print('-'*10)
    filepath='/home/guozy/Ni-YSZ_CNN/model/'
    subpath = os.listdir(filepath)
    count = -1
    for p in subpath:
        count += 1
        print('{},{}'.format(count, p))
    print('-' * 10)

    choice=int(input('input the number of the determined model\n'))
    filepath=filepath+subpath[choice]+'/model.h5'
    print('load model')
    model = load_model(filepath,compile=False)  # 根据具体读取模型地址修改

    return model


# 在测试集上评估模型
def pred_test(model,test_x,test_y):
    # Predict
    pred_y = np.array(model.predict(test_x),dtype='float64')
    max_min_nor = np.array([[1.65219166e+05,4.88838187e+04],[2.84213381e-01,2.07984799e-01],[6.37954456e+04,1.68850886e+04],[1.58147200e-05,1.19209400e-05],[1.34268741e+01,4.93731340e+00]])

    pred_y[:,0]=pred_y[:,0]*(max_min_nor[0,0]-max_min_nor[0,1])+max_min_nor[0,1]
    pred_y[:,1]=pred_y[:,1]*(max_min_nor[1,0]-max_min_nor[1,1])+max_min_nor[1,1]
    pred_y[:,2]=pred_y[:,2]*(max_min_nor[2,0]-max_min_nor[2,1])+max_min_nor[2,1]
    pred_y[:,3]=pred_y[:,3]*(max_min_nor[3,0]-max_min_nor[3,1])+max_min_nor[3,1]
    pred_y[:,4]=pred_y[:,4]*(max_min_nor[4,0]-max_min_nor[4,1])+max_min_nor[4,1]

    r2=np.zeros([5])
    mae=np.zeros([5])
    mape=np.zeros([5])
    wmape=np.zeros([5])
    for i in range(5):
        pred=pred_y[:,i]
        test=test_y[:,i]
    
        r2[i]=r2_score(pred,test)
        mae[i]=mean_absolute_error(pred,test)
        mape[i]=np.mean(np.abs((pred-test)/test))*100
        wmape[i]=np.sum(np.abs(pred-test))*100
        wmape[i]=wmape[i]/np.sum(np.abs(test))
        
    wmape_all=np.sum(np.abs(pred_y-test_y))*100/np.sum(np.abs(test_y))

    print("r2:",r2)
    print("mae:",mae)
    print("mape:",mape)
    print("wmape:",wmape)



if __name__=='__main__':
    
    test_x,test_y=load_data()
    model = get_model()
    pred_test(model,test_x,test_y)
    test_x,test_y=aug_data(test_x,test_y)
    pred_test(model,test_x,test_y)




