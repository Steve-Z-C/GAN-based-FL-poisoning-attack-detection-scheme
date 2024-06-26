from __future__ import print_function

import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
import numpy as np
import utils

filepath = ""

def f_normal_data_create(i):
    #加载mnist数据集，x/y_train_0为原始训练集，x/y_train_1~i为客户端训练集
    x_train, y_train = utils.load_mnist('./dataset/fashion', kind='train')
    x_test, y_test = utils.load_mnist('./dataset/fashion', kind='t10k')
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