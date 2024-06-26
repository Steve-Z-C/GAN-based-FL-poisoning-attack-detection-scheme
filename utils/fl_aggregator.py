from __future__ import print_function
import tensorflow as tf
# import keras
from tensorflow.keras import backend as K
from tensorflow.keras import datasets, layers, models, optimizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

from sklearn.metrics import accuracy_score, f1_score

from utils import *
import numpy as np
from glob import glob

# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# set_session(tf.Session(config=config))

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def load_models(client_model_path, round):
    arr = []
    models = glob(client_model_path + r"/client_model_" + str(round) + r"_" + r"*.h5")
    models = sorted(models, key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))
    for i in models:
        print(i)
        model = load_model(i)
        md1 = model.get_weights()
        arr.append(md1)
    return np.array(arr)

def fl_average(client_model_path, round):
    # FL average
    arr = load_models(client_model_path, round)
    fl_avg = np.average(arr, axis=0)
    # fl_avg = np.average(arr)
    for i in fl_avg:
        print(i.shape)
    return fl_avg

def build_model(avg):

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.SGD(lr=0.001, decay=0.0, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])

    model.set_weights(avg)

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.SGD(lr=0.001, decay=0.0, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])

    return model

def build_cifar_model(avg):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     input_shape=(32, 32, 3),
                     activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.SGD(lr=0.01, decay=0.0, momentum=0.0, nesterov=True),
                  metrics=['accuracy'])

    model.set_weights(avg)
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.SGD(lr=0.01, decay=0.0, momentum=0.0, nesterov=True),
                  metrics=['accuracy'])
    return model

def evaluate_agg_model(model, x_test, y_test, accpath, losspath):
    score = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    acc = open(accpath, "a+")
    acc.write(str(score[1]) + "\n")
    acc.close()
    loss = open(losspath, "a+")
    loss.write(str(score[0]) + "\n")
    loss.close()

def save_agg_model(model, path, round):
    global_model_path = path + r"/global_model" + "_"+ str(round) + ".h5"
    model.save(global_model_path)
    print('global_model has been saved in: ' +  global_model_path)

def model_aggregation(x_test, y_test, global_model_path, client_model_path, round, accpath, losspath):
    avg = fl_average(client_model_path, round)
    model = build_model(avg)
    evaluate_agg_model(model, x_test, y_test, accpath, losspath)
    save_agg_model(model, global_model_path, round)
    print("The " + str(round) + " round has finished!")

def poisoning_determination(global_model, client_model, x_test: np.ndarray, y_test: np.ndarray, f1_loss, acc_loss):
    y_pred_global = global_model.predict(x_test)
    y_pred_client = client_model.predict(x_test)
    f1_global = f1_score(y_test, y_pred_global, average='weighted')
    f1_client = f1_score(y_test, y_pred_client, average='weighted')
    score_0 = global_model.evaluate(x_test, y_test, verbose=1)
    acc_global = score_0[1]
    score_1 = client_model.evaluate(x_test, y_test, verbose=1)
    acc_client = score_1[1]
    if (f1_client - f1_global) < f1_loss and (acc_client - acc_global) < acc_loss:
        return True
    else:
        return False

def safety_determination(client_model_path, global_model_path, round, x_test, y_test, f1_loss, acc_loss, safety_matrix, safety_param):
    arr = load_models(client_model_path, round)
    global_model_path = global_model_path + r"/global_model_"+ str(round-1) + ".h5"
    global_model = load_model(global_model_path, custom_objects=None, compile=True)
    for i in range(len(arr)):
        if not poisoning_determination(global_model, arr[i], x_test, y_test, f1_loss, acc_loss):
            safety_matrix[i] += 1
        if not safety_matrix[i] < safety_param:
            del arr[i]
    # 返回safety_matrix，后续删除第i个恶意客户端的训练集
    return arr, safety_matrix

# 比较F-1损失/acc损失，统计一个整型数组作为客户端标记次数，与安全参数进行比较
def model_detect_aggregation(x_test, y_test, global_model_path, client_model_path, round, accpath, losspath, f1_loss, acc_loss, safety_matrix, safety_param):
    arr, safety_matrix = safety_determination(client_model_path, global_model_path, round, x_test, y_test, f1_loss, acc_loss, safety_matrix, safety_param)
    fl_avg = np.average(arr, axis=0)
    for i in fl_avg:
        print(i.shape)
    model = build_model(fl_avg)
    evaluate_agg_model(model, x_test, y_test, accpath, losspath)
    save_agg_model(model, global_model_path, round)
    print("The " + str(round) + " round has finished!")
    return safety_matrix