from __future__ import print_function

import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist

from sklearn.model_selection import train_test_split

import numpy as np

import os

filepath = ""

def mnist_noniid(i):
    #加载mnist数据集，x/y_train_0为原始训练集，x/y_train_1~i为客户端训练集
    path = r"./dataset/mnist.npz"
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path)
    #处理原始数据集格式并保存
    x_train_0,x_train_else,y_train_0,y_train_else = train_test_split(x_train, y_train, test_size=0.5, random_state=0)
    x_test_0,x_test_1,y_test_0,y_test_1 = train_test_split(x_test, y_test, test_size=0.5, random_state=0)
    np.save(filepath + "/x_train_0.npy", x_train_0)
    np.save(filepath + "/y_train_0.npy", y_train_0)
    np.save(filepath + "/x_test_0.npy", x_test_0)
    np.save(filepath + "/y_test_0.npy", y_test_0)
    np.save(filepath + "/x_test_1.npy", x_test_1)
    np.save(filepath + "/y_test_1.npy", y_test_1)    
    if i > 0:
        while i > 1:
            x_train_i,x_train_else,y_train_i,y_train_else = train_test_split(x_train_else, y_train_else, test_size=(i-1)/i, random_state=0)
            np.save(filepath + "/x_train_" + str(i) + r".npy", x_train_i)
            np.save(filepath + "/y_train_" + str(i) + r'.npy', y_train_i)
            i -= 1
    else:
        print("Client error!")

    np.save(filepath + "/x_train_"+ str(1) + r".npy", x_train_else)
    np.save(filepath + "/y_train_"+ str(1) + r".npy", y_train_else)

def data_processing(x, y):
    if K.image_data_format()=='channels_first':
        x= x.reshape(-1,1,28,28)
    else:
        x = x.reshape(-1,28,28,1)
    #数据归一化，将样本转化为浮点数
    x = x / 255.0
    #将标签转化为onehot编码
    y = tf.keras.utils.to_categorical(y,10)
    return x, y