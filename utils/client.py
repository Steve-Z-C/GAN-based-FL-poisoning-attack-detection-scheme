from __future__ import print_function
from tensorflow.keras.applications.vgg16 import VGG16
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers, applications
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
# from keras import losses
from os import path
import numpy as np
import glob
import h5py
import utils

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

first_model_path = './models(additional_experiment)/first_model.h5'

def train_client_model(x_train, y_train, x_test, y_test, round, epoch, b_size, learning_rate, global_model_path):
    # if there is model update set its weights as model weights
    global_model_path = global_model_path + r"/global_model_"+ str(round-1) + ".h5"
    if path.exists(global_model_path):
        print("Global model exists...\nLoading model...")
        model = load_model(global_model_path, custom_objects=None, compile=True)
    elif path.exists(first_model_path):
        print("First model exists...\nLoading model...")
        model = load_model(first_model_path, custom_objects=None, compile=True)
        
    else:
        print("No model found!\nBuilding model...")

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
        # model.summary()

        model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.SGD(lr=learning_rate, decay=0.0, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
            optimizer=tf.keras.optimizers.SGD(lr=learning_rate, decay=0.0, momentum=0.9, nesterov=True),
            metrics=['accuracy'])
    
    model.fit(x_train, y_train, batch_size=b_size, epochs=epoch, verbose=1, validation_data=(x_test,y_test))

    return model

def model_random_attack(x_train, y_train, x_test, y_test, round, epoch, b_size, learning_rate, global_model_path):
    global_model_path = global_model_path + r"/global_model_"+ str(round-1) + ".h5"
    if path.exists(global_model_path):
        print("Global model exists...\nLoading model...")
        model = load_model(global_model_path, custom_objects=None, compile=True)
    elif path.exists(first_model_path):
        print("First model exists...\nLoading model...")
        model = load_model(first_model_path, custom_objects=None, compile=True)
        
    else:
        print("No model found!\nBuilding model...")

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
        # model.summary()

        model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.SGD(lr=learning_rate, decay=0.0, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
            optimizer=tf.keras.optimizers.SGD(lr=learning_rate, decay=0.0, momentum=0.9, nesterov=True),
            metrics=['accuracy'])
    
    model.fit(x_train, y_train, batch_size=b_size, epochs=epoch, verbose=1, validation_data=(x_test,y_test))
    
    md1 = model.get_weights()

    for i in range(len(md1)):
        m = md1[i].flatten()
        noise = np.random.normal(0, 0.05, len(m)) # random noise
        m += noise
        md1[i] = m

    md1[0] = np.array(md1[0]).reshape(3, 3, 1, 32)
    md1[1] = np.array(md1[1]).reshape(32,)
    md1[2] = np.array(md1[2]).reshape(3, 3, 32, 64)
    md1[3] = np.array(md1[3]).reshape(64,)
    md1[4] = np.array(md1[4]).reshape(9216, 128)
    md1[5] = np.array(md1[5]).reshape(128,)
    md1[6] = np.array(md1[6]).reshape(128, 10)
    md1[7] = np.array(md1[7]).reshape(10,)
    # md1 = list(md1.values())

    model2 = Sequential()
    model2.add(Conv2D(32, kernel_size=(3, 3),
                        activation='relu',
                        input_shape=(28,28,1)))
    model2.add(Conv2D(64, (3, 3), activation='relu'))

    model2.add(MaxPooling2D(pool_size=(2, 2)))
    model2.add(Dropout(0.25))
    model2.add(Flatten())
    model2.add(Dense(128, activation='relu'))
    model2.add(Dropout(0.5))
    model2.add(Dense(10, activation='softmax'))

    model2.compile(loss=tf.keras.losses.categorical_crossentropy,
                    optimizer=tf.keras.optimizers.SGD(lr=learning_rate, decay=0.0, momentum=0.9, nesterov=True),
                    metrics=['accuracy'])
    model2.set_weights(md1)
    model2.compile(loss=tf.keras.losses.categorical_crossentropy,
                    optimizer=tf.keras.optimizers.SGD(lr=learning_rate, decay=0.0, momentum=0.9, nesterov=True),
                    metrics=['accuracy'])

    return model2



#测试精度
def evaluate_client_model(model, x_test: np.ndarray, y_test: np.ndarray):
    score = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

#保存更新后的模型
def save_model_update(model, model_num, round, path):
    update_model_path = path + "/client_model_" + str(round) + "_" + str(model_num) + ".h5"
    model.save(update_model_path)
    print('client_model_' + str(model_num) + ' has been saved in: ' + update_model_path)

