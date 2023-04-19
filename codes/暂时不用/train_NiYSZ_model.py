import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.layers import Input, Flatten, Convolution3D, AveragePooling3D, Dropout
from keras.callbacks import EarlyStopping
from keras.callbacks import Callback
import tensorflow as tf
from tensorflow.python.framework import ops
from keras.regularizers import l2
from PIL import Image
from matplotlib import pyplot as plt
import time
import os
from keras.initializers import glorot_normal
from keras.optimizers import adam_v2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print ('load data')

# read the tifs
final_train_data=np.zeros([1347,64,64,64,1])

tiff_num=[5000,10000,20000,30000];

count=-1;

for cycleT in range(4):
    
    for cycleTiff in range(357):
        filename = "/home/guozy/Ni-YSZ_CNN/train/anode_stacks_" + str(tiff_num[cycleT]) + '/' + str(cycleTiff + 1) + ".tiff"
    
        if os.path.exists(filename)==False:
            continue
        
        count=count+1;
        
        img = Image.open(filename)    
    
        for i in range(img.n_frames):    #可以用img.n_frames来得到总页数
            try:
                img.seek(i)    #是否存在
                final_train_data[count,:,:,i,0]=img;
            except EOFError:  #页数读完了
                break

#让 0 127 255 变成 -1 0 1
final_train_data[final_train_data==0]=-1
final_train_data[final_train_data==127]=0
final_train_data[final_train_data==255]=1

# 画图
#plt.imshow(final_train_data[1,:,:,0,0]) 

# read label data from txt
with open("/home/guozy/Ni-YSZ_CNN/train/Results_all.txt", "r") as f:
    line = f.readline()
    label_list = []
    while line:
        num = list(map(float,line.split()))
        label_list.append(num)
        line = f.readline()
    f.close()
    final_train_label = np.array(label_list)
  
train_max = np.max(final_train_label, axis=0)
train_min = np.min(final_train_label, axis=0)
#
for i in range(final_train_label.shape[1]):
    final_train_label[:, i] = (final_train_label[:, i] - train_min[i]) / (train_max[i] - train_min[i])



#把label的最大最小值列出并保存在txt，在与预测调用

print ('training data shape: ', final_train_data.shape)
print ('training label shape: ', final_train_label.shape)

batchsize=[20]
#batchsize=[5]

#卷积核
kernel=3

for i in range(1):

    time_start=time.time()
    
    print ('create model')
    model = Sequential()
    
    #第一层卷积
    #三维卷积对三维的输入进行滑动窗卷积，当使用该层作为第一层时，应提供 input_shape 参数
    #Convolution3D(卷积核的数目，卷积核3个维度的长，init：初始化方法，border_mode：边界模式，W_regularizer：施加在权重上的正则项)
    model.add(Convolution3D(16, (kernel, kernel, kernel), kernel_initializer=glorot_normal(), padding='same', kernel_regularizer=l2(0.001), input_shape=(64,64,64,1)))
    #激活层：对一个层的输出施加激活函数，输出与输入shape相同，采用 relu 激活层
    model.add(Activation('relu'))
    #平均值池化层：降维
    model.add(AveragePooling3D(pool_size=(2, 2, 2)))
    
    #第二层卷积
    model.add(Convolution3D(32,(kernel, kernel, kernel), kernel_initializer=glorot_normal(), padding='same', kernel_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(AveragePooling3D(pool_size=(2, 2, 2)))
    
    #第三层卷积
    model.add(Convolution3D(64, (kernel, kernel, kernel), kernel_initializer=glorot_normal(), padding='same', kernel_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(AveragePooling3D(pool_size=(2, 2, 2)))
    
    #第四层卷积
    model.add(Convolution3D(128, (kernel, kernel, kernel), kernel_initializer=glorot_normal(), padding='same', kernel_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(AveragePooling3D(pool_size=(2, 2, 2)))
    
    #第五层卷积
    model.add(Convolution3D(256, (kernel, kernel, kernel), kernel_initializer=glorot_normal(), padding='same', kernel_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(AveragePooling3D(pool_size=(2, 2, 2)))
    
    #第六层卷积
    model.add(Convolution3D(512, (kernel, kernel, kernel), kernel_initializer=glorot_normal(), padding='same', kernel_regularizer=l2(0.001)))
    model.add(Activation('relu'))
    model.add(AveragePooling3D(pool_size=(2, 2, 2)))
    
    # #第七层卷积
    # model.add(Convolution3D(512, kernel, kernel, kernel, init='glorot_normal', border_mode='same', W_regularizer=l2(0.001)))
    # model.add(Activation('relu'))
    # model.add(AveragePooling3D(pool_size=(2, 2, 2)))
    
    #Flatten层：用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响 batch的大小。
    model.add(Flatten())
    
    #keras.layers.concatenate
    
    # Dense：常用的全连接层
    # Dense(该层的输出维度，init：初始化方法，activation：激活函数，W_regularizer：施加在权重上的正则项)
    #model.add(Dense(2048, init='glorot_normal', activation='relu', W_regularizer=l2(0.001)))
    #model.add(Dense(2048, init='glorot_normal', activation='relu', W_regularizer=l2(0.001)))
    model.add(Dense(2048, kernel_initializer=glorot_normal(), activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(1024, kernel_initializer=glorot_normal(), activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(5, kernel_initializer=glorot_normal(),kernel_regularizer=l2(0.001)))
    
    print ('compile model')
    print (model.summary())
    
    #编译用来配置模型的学习过程 
    # compile(loss：字符串（预定义损失函数名）或目标函数，optimizer：字符串（预定义优化器名）或优化器对象，metrics：列表，包含评估模型在训练和测试时的网络性能的指标)
    model.compile(loss='mae', optimizer=adam_v2.Adam(learning_rate=1e-5), metrics=['mae','mse','mape'])
    
    print ('-------------------------')
    print ('fit model')
    
    # 当监测值不再改善时，该回调函数将中止训练
    # EarlyStopping(monitor：需要监视的量；patience：当early stop被激活（如发现loss相比上一个epoch训练没有下降），则经过patience个epoch后停止训练。)
    #early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    
    #训练模型
    # fit(batch_size：整数，指定进行梯度下降时每个batch包含的样本数；nb_epoch：整数，训练的轮数，训练数据将会被遍历 nb_epoch次；
    #validation_split：0~1之间的浮点数，用来指定训练集的一定比例数据作为验证集；callbacks：list，其中的元素是keras.callbacks.Callback的对象。这个list中的回调函数将会在训练过程中的适当时机被调用)
    history = model.fit(final_train_data, final_train_label, batch_size=batchsize[i], epochs=1000, validation_split=0.1)
    
    time_end=time.time()
    print('\ntotally cost:\n',time_end-time_start)
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    mse = history.history['mse']
    val_mse = history.history['val_mse']
    mape = history.history['mape']
    val_mape = history.history['val_mape']
  
    # Output to the txt
    txtfile=open('/home/guozy/Ni-YSZ_CNN/model/1000.txt','w+')
    
    length=len(loss)
    
    for cyclelen in range(int(length/500)):
    
        loc=500*(cyclelen+1)
        
        average=np.zeros(6)
        for cycleresult in range(10):
            average[0]=average[0]+loss[loc-1-cycleresult]
            average[1]=average[1]+val_loss[loc-1-cycleresult]
            average[2]=average[2]+mse[loc-1-cycleresult]
            average[3]=average[3]+val_mse[loc-1-cycleresult]
            average[4]=average[4]+mape[loc-1-cycleresult]
            average[5]=average[5]+val_mape[loc-1-cycleresult]   
        
        txtfile.write(str(average[0]/10)+'\t'+str(average[2]/10)+'\t'+str(average[4]/10)+'\t'+str(average[1]/10)+'\t'+str(average[3]/10)+'\t'+str(average[5]/10)+'\n')
        
    txtfile.write(str(time_end-time_start)+'\n')
    
    # Close the txt
    txtfile.close()
           


print ('save model')
model.save('/home/guozy/Ni-YSZ_CNN/model/1000.h5')


