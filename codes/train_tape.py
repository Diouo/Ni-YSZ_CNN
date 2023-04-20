import random
import time
import os

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from contextlib import redirect_stdout

import tensorflow as tf
import keras

from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.layers import Input, Flatten, Convolution3D, AveragePooling3D, Dropout, MaxPooling3D,BatchNormalization
from keras.initializers import glorot_normal
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model
from keras.optimizers import adam_v2
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split


random.seed(42)
local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
filepath = '/home/guozy/Ni-YSZ_CNN/model/' + local_time.replace(' ', '_') + '/'
folder = os.path.exists(filepath)
if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
    os.makedirs(filepath)  # makedirs 创建文件时如果路径不存在会创建这个路径
plt.switch_backend('agg')
summary_writer=tf.summary.create_file_writer('./tensorboard')


# 载入训练集和测试集的数据
def load_data():
    print('load data\n')  # 服务器进度展示

    # 载入训练集的x
    tiff_num = [5000, 10000, 20000, 30000]  # 4个文件夹的名字
    count = -1  # 用来记录打开了第几个样本了
    train_x = np.zeros([1347, 64, 64, 64, 1])  # 训练集的x
    for cycleT in range(4):
        for cycleTiff in range(357):
            filename = "/home/guozy/Ni-YSZ_CNN/train/anode_stacks_" + str(tiff_num[cycleT]) + '/' + str(
                cycleTiff + 1) + ".tiff"
            if os.path.exists(filename) == False:  # 找不到就跳过，因为文件里有一些名字带no
                continue
            count = count + 1
            img = Image.open(filename)
            for i in range(img.n_frames):  # 可以用img.n_frames来得到总页数
                try:
                    img.seek(i)  # 是否存在
                    train_x[count, :, :, i, 0] = img
                except EOFError:  # 页数读完了
                    break

    # 再载入训练集的y
    with open("/home/guozy/Ni-YSZ_CNN/train/Results_all.txt", "r") as f:
        line = f.readline()
        label_list = []
        while line:
            num = list(map(float, line.split()))
            label_list.append(num)
            line = f.readline()
        f.close()
        train_y = np.array(label_list)

    # 载入测试集的x
    count = -1
    test_x = np.zeros([344, 64, 64, 64, 1])
    for cycleTiff in range(357):
        filename = "/home/guozy/Ni-YSZ_CNN/predict/anode_stacks_15000/" + str(cycleTiff + 1) + ".tiff"
        if os.path.exists(filename) == False:
            continue
        count = count + 1
        img = Image.open(filename)
        for i in range(img.n_frames):  # 可以用img.n_frames来得到总页数
            try:
                img.seek(i)  # 是否存在
                test_x[count, :, :, i, 0] = img
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
        test_y = np.array(label_list)

    print('training x shape: ', train_x.shape)
    print('training y shape: ', train_y.shape)
    print('testing x shape: ', test_x.shape)
    print('testing y shape: ', test_y.shape)
    print('\n')

    return train_x, train_y, test_x, test_y


# 数据预处理
def process(train_x, train_y, test_x, test_y):
    print('process data\n')

    # y归一化到[0,1]
    train_max = np.max(train_y, axis=0)
    train_min = np.min(train_y, axis=0)
    #
    for i in range(train_y.shape[1]):
        train_y[:, i] = (train_y[:, i] - train_min[i]) / (train_max[i] - train_min[i])
        test_y[:, i] = (test_y[:, i] - train_min[i]) / (train_max[i] - train_min[i])

    # 保存训练集信息
    file = open('/home/guozy/Ni-YSZ_CNN/train/max_min.txt', 'w')
    file.write('train_max:' + str(train_max) + '\n')
    file.write('train_max:' + str(train_min) + '\n')
    file.close()

    return train_x, train_y, test_x, test_y


# 数据增强中的图像旋转
def rotate(xi, choice):
    x_aug = np.zeros([1, 64, 64, 64, 1])  # 增强后的样本x

    # 分9种情况进行讨论
    if choice == 1:
        for i in range(64):
            x_aug[0, i, :, :, 0] = np.rot90(xi[i, :, :], 1)

    elif choice == 2:
        for i in range(64):
            x_aug[0, i, :, :, 0] = np.rot90(xi[i, :, :], 2)

    elif choice == 3:
        for i in range(64):
            x_aug[0, i, :, :, 0] = np.rot90(xi[i, :, :], 3)

    elif choice == 4:
        for i in range(64):
            x_aug[0, :, i, :, 0] = np.rot90(xi[:, i, :], 1)

    elif choice == 5:
        for i in range(64):
            x_aug[0, :, i, :, 0] = np.rot90(xi[:, i, :], 2)

    elif choice == 6:
        for i in range(64):
            x_aug[0, :, i, :, 0] = np.rot90(xi[:, i, :], 3)

    elif choice == 7:
        for i in range(64):
            x_aug[0, :, :, i, 0] = np.rot90(xi[:, :, i], 1)

    elif choice == 8:
        for i in range(64):
            x_aug[0, :, :, i, 0] = np.rot90(xi[:, :, i], 2)

    elif choice == 9:
        for i in range(64):
            x_aug[0, :, :, i, 0] = np.rot90(xi[:, :, i], 3)

    return x_aug


# 镜像
def mirror(xi,choice):
    choice -= 9  # 选择镜像的面
    x_aug = np.zeros([1, 64, 64, 64, 1])  # 增强后的样本x

    # 分9种情况进行讨论
    if choice == 1 or choice == 5:  # x轴为对称轴
        for i in range(64):
            x_aug[0, :, :, i, 0] = np.flip(xi[:, :, i], axis=0)

    elif choice == 2 or choice == 3: # y轴为对称轴
        for i in range(64):
            x_aug[0, :, :, i, 0] = np.flip(xi[:, :, i], axis=1)

    elif choice == 4 or choice == 6: # z轴为对称轴
        for i in range(64):
            x_aug[0, i, :, :, 0] = np.flip(xi[i, :, :], axis=1)

    return x_aug


# 数据增强
def aug_data(x, y):
    print('augment data\n')
    total = x.shape[0]  # 总共有多少幅图
    stat = np.zeros([15])

    for i in range(total):
        xi = x[i, :, :, :, 0]  # 得到第i个样本
        choice =  random.randint(1,15) # 选择：翻转/镜像
        stat[choice - 1] = stat[choice - 1] + 1
        if choice <= 9:
            x_aug = rotate(xi, choice)  
        else:
            x_aug = mirror(xi, choice)
        # 将数据增强后的样本和原来的样本合并
        x = tf.concat([x, x_aug], axis=0)

    y = tf.concat([y, y], axis=0)
    print(stat,'\n')

    with open(filepath + 'information.txt', 'a') as f:
        f.write('augment train data\n\n')
        f.close()

    return x.numpy(), y.numpy()


# 生成并返回模型
def gen_model():
    print('generate model\n')  # 进度展示
    model = Sequential()

    # 卷积核的边长
    kernel = 3

    # 第0层卷积
    # 三维卷积对三维的输入进行滑动窗卷积，当使用该层作为第一层时，应提供 input_shape 参数
    # Convolution3D(卷积核的数目，卷积核3个维度的长，kernel_regularizer：权重的初始化方法，padding：边界模式，kernel_regularizer：权重上的正则项)
    model.add(Convolution3D(16, (kernel, kernel, kernel), kernel_initializer=glorot_normal(seed=42), padding='same',input_shape=(64, 64, 64, 1), kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling3D(pool_size=(2, 2, 2), data_format='channels_last'))

    # 第1层卷积
    model.add(Convolution3D(32, (kernel, kernel, kernel), kernel_initializer=glorot_normal(seed=42), padding='same',kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling3D(pool_size=(2, 2, 2), data_format='channels_last'))

    # 第2层卷积
    model.add(Convolution3D(64, (kernel, kernel, kernel), kernel_initializer=glorot_normal(seed=42), padding='same',kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling3D(pool_size=(2, 2, 2), data_format='channels_last'))

    # 第3层卷积
    model.add(Convolution3D(128, (kernel, kernel, kernel), kernel_initializer=glorot_normal(seed=42), padding='same',kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling3D(pool_size=(2, 2, 2), data_format='channels_last'))

    # 第4层卷积
    model.add(Convolution3D(256, (kernel, kernel, kernel), kernel_initializer=glorot_normal(seed=42), padding='same',kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling3D(pool_size=(2, 2, 2), data_format='channels_last'))

    # 第5层卷积
    model.add(Convolution3D(512, (kernel, kernel, kernel), kernel_initializer=glorot_normal(seed=42), padding='same',kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling3D(pool_size=(2, 2, 2), data_format='channels_last'))

    # Flatten层：用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。
    model.add(Flatten())

    # Dense：常用的全连接层
    # Dense(该层的输出维度，init：初始化方法，activation：激活函数，kernel_regularizer：施加在权重上的正则项)
    model.add(Dense(2048, activation='relu', kernel_initializer=glorot_normal(seed=42), kernel_regularizer=l2(0.001)))
    model.add(Dense(1024, activation='relu', kernel_initializer=glorot_normal(seed=42), kernel_regularizer=l2(0.001)))
    model.add(Dense(5, kernel_initializer=glorot_normal(seed=42),kernel_regularizer=l2(0.001)))  # 设定为训练集y的列数，增加了鲁棒性

    print(model.summary(), '\n')
    with open(filepath + 'information.txt', 'a') as f:
        with redirect_stdout(f):
            model.summary()

    return model


if __name__ == '__main__':
    # 提前确定好的超参数
    learning_rate = 1e-5
    batchsize = 20
    epochs = 1000
    validation_split = 0.1

    # 数据相关
    train_x, train_y, test_x, test_y = load_data()  # 载入训练集和数据集的数据
    train_x, train_y, test_x, test_y = process(train_x, train_y, test_x, test_y)  # 数据预处理
    # 划分测试集和验证集
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y,test_size=validation_split,random_state=42)
    # 数据增强
    train_x, train_y = aug_data(train_x, train_y)  # 做训练集数据增强
    # 生成数据库
    train_db = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    train_db =train_db.batch(batchsize).shuffle(1000)
    train_iter=iter(train_db)
    
    # 模型训练相关
    time_start = time.time()
    model = gen_model()  # 生成模型
    optimizer = adam_v2.Adam(learning_rate=learning_rate,beta_1=0.999)
    variables = model.trainable_variables

    # 可视化
    TRAIN_loss = []
    VAL_loss = []

    for epoch in range(epochs):
        loss_average = 0
        for step, (x, y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                out = model(x,training=True)
                loss = tf.reduce_mean(keras.losses.MAE(y, out))
                loss_average = loss_average + loss

            grads = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(grads, variables))
        TRAIN_loss.append(loss_average/step)
        

        if epoch % 5==0:
            out = model(val_x, training=False)
            y = tf.cast(val_y, dtype=tf.float32)
            val_loss = tf.reduce_mean(keras.losses.MAE(y, out))
            VAL_loss.append(val_loss)
            print('epoch:{}, train_loss:{}, val_loss'.format(epoch, loss_average/step, val_loss))
        else:
            print('epoch:{}, train_loss:{}'.format(epoch, loss_average/step))



    time_end = time.time()
    print('\ntotally cost:', time_end - time_start, '\n')  # 查看一次训练需要花多少时间
    with open(filepath + 'information.txt', 'a') as f:
        f.write('totally cost:{}\n\n'.format(time_end - time_start))
        f.close()

    # 模型结果相关
    print('save model and related information\n')
    model.save(filepath + 'model.h5')  # 保存模型

    # 保存损失
    TRAIN_loss = np.array(TRAIN_loss)
    VAL_loss = np.array(VAL_loss)
    np.savetxt(filepath + "TRAIN_loss.txt", TRAIN_loss , fmt="%d", delimiter=",")
    np.savetxt(filepath +"VAL_loss.txt", VAL_loss , fmt="%d", delimiter=",")


    # 删除模型
    print('delete model\n')
    del model




