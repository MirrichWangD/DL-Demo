# -*- coding: utf-8 -*-

"""
第05讲 卷积神经网络识别CIFAR-10图像
VGG13 网络识别 CIFAR-10数据集
@ 参考龙老师书:p248页第十章 实战案例
======================================="""

#导入TensorFlow模块，keras作为tensorflow的子模块: tf.keras
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf



""""@@@ 数据块 """
#导入CIFAR10图像数据
cifar10 = tf.keras.datasets.cifar10
(X_img_train, y_label_train), (X_img_test, y_label_test) = cifar10.load_data()

# 数据归一化
X_img_train_normalize = X_img_train.astype('float32') / 255.0
X_img_test_normalize = X_img_test.astype('float32') / 255.0

# 标签OneHot化
y_label_train_OneHot = tf.keras.utils.to_categorical(y_label_train)
y_label_test_OneHot = tf.keras.utils.to_categorical(y_label_test)



""""@@@ 网络块 VGG13 """
# 建立卷积神经网络模型（VGG13网络）
# 注意：原输入224X224，现调整为：32X32。 输出为1000，调整为：10
model = tf.keras.Sequential()

#卷积层1与池化层1
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), input_shape=(32, 32, 3),
                                 activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3),
                                 activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))


#卷积层2与池化层2
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3),
                                 activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3),
                                 activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))


#卷积层3与池化层3
model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3),
                                 activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3),
                                 activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))


#卷积层4与池化层4
model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3),
                                 activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3),
                                 activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))


#卷积层5与池化层5
model.add(tf.keras.layers.Conv2D(filters=512,kernel_size=(3, 3),
                                 activation='relu', padding='same'))
model.add(tf.keras.layers.Conv2D(filters=512,kernel_size=(3, 3),
                                 activation='relu', padding='same'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))


#建立神经网络(平坦层、隐藏层、输出层)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(4096, activation='relu'))
model.add(tf.keras.layers.Dense(4096, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

#输出模型列表，需要调整近3000万的参数，Total params: 28,328,522
print(model.summary())



""""@@@ 训练模型 """
epochs=10   #时间：2分钟  训练集准确率：9.87%   测试集准确率：10.16%
# epochs=50 #时间：分钟  训练集准确率：%   测试集准确率：%
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

t1=time.time()
train_history=model.fit(X_img_train_normalize, y_label_train_OneHot,
                        validation_split=0.2,
                        epochs=epochs, batch_size=128, verbose=1)
t2=time.time()
CNNVGG13 = float(t2-t1)
print("Time taken: {} seconds".format(CNNVGG13))



""""@@@ 评估模型的准确率 """
def show_train_history(train_acc,test_acc):
    plt.plot(train_history.history[train_acc])
    plt.plot(train_history.history[test_acc])
    plt.title('Train History')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

show_train_history('accuracy','val_accuracy') 
show_train_history('loss','val_loss')

# 测试集准确率：10.00%
scores = model.evaluate(X_img_test_normalize, 
                        y_label_test_OneHot,verbose=0)
scores[1]



""""@@@ 进行预测 """
prediction=np.argmax(model.predict(X_img_test_normalize),axis=1)

label_dict={0:"airplane",1:"automobile",2:"bird",3:"cat",4:"deer",
            5:"dog",6:"frog",7:"horse",8:"ship",9:"truck"}


def plot_images_labels_prediction(images,labels,prediction,
                                  idx,num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num>25: num=25 
    for i in range(0, num):
        ax=plt.subplot(5,5, 1+i)
        ax.imshow(images[idx],cmap='binary')
                
        title=str(i)+','+label_dict[labels[i][0]]
        if len(prediction)>0:
            title+='=>'+label_dict[prediction[i]]
            
        ax.set_title(title,fontsize=10) 
        ax.set_xticks([]);ax.set_yticks([])
        idx+=1 
    plt.show()
    
plot_images_labels_prediction(X_img_test_normalize,y_label_test,
                              prediction,0,10)


# 查看预测概率
Predicted_Probability=model.predict(X_img_test_normalize)

def show_Predicted_Probability(X_img,Predicted_Probability,i):
    plt.figure(figsize=(2,2))
    plt.imshow(np.reshape(X_img_test[i],(32, 32,3)))
    plt.show()
    for j in range(10):
        print(label_dict[j]+' Probability:%1.9f'%(Predicted_Probability[i][j]))


show_Predicted_Probability(X_img_test,Predicted_Probability,0)


""""@@@ save model 储存模型结构和权重 """
model_json = model.to_json()
open('18DL/SaveModel/VGG13_architecture.json', 'w').write(model_json)
model.save_weights('18DL/SaveModel/VGG13_weights.h5', overwrite=True)

