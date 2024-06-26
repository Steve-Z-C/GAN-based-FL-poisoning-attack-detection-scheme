from __future__ import print_function
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers, applications, regularizers
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LeakyReLU, BatchNormalization, Activation
from tensorflow.keras import backend as K
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.utils import to_categorical



first_model_path = r'./models/new/global model/models/first_model.h5'


def train_global_model10(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray):
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
    model.summary()
    
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.SGD(lr=0.01, decay=0.0, momentum=0.0, nesterov=True),
                  metrics=['accuracy'])
    
    model.fit(x_train, y_train, batch_size=128, epochs=10, verbose=1, validation_data=(x_test, y_test))

    # model = load_model(first_model_path, custom_objects=None, compile=True)
    # model = Sequential()
    
    # model.add(Conv2D(64, kernel_size=(3, 3),
    #                  activation='relu',
    #                  padding='same',
    #                  input_shape=(32, 32, 3)))
    # model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    # model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    # model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    # model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    # model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    # model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    # model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    # model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Flatten())
    # model.add(Dense(4096, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(4096, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(10, activation='softmax'))
    # model.summary()

    # ishape = 32
    # tf.keras.initializers.glorot_normal(seed=None)
    # model_vgg = VGG16(include_top=False, weights='imagenet',  input_shape=(ishape, ishape, 3))
    # for layers in model_vgg.layers:
    #     layers.trainable = False
    # model = Flatten()(model_vgg.output)
    # model = Dense(4096, activation='relu', name='fc1')(model)
    # model = Dropout(0.25)(model)
    # model = Dense(4096, activation='relu', name='fc2')(model)
    # model = Dropout(0.25)(model)
    # model = Dense(10, activation='softmax', name='prediction')(model)
    # model = Model(inputs=model_vgg.input, outputs=model, name='vgg16_pretrain')
    # model.summary()

    # model.compile(
    #     loss='categorical_crossentropy',
    #     optimizer=optimizers.SGD(lr=0.01, decay=0.0, momentum=0.0, nesterov=True),
    #     metrics=['accuracy'])  # 损失函数为分类交叉熵，优化器为SGD


    # model.fit(
    #     x=x_train,
    #     y=y_train,
    #     batch_size=128,
    #     epochs=20,
    #     verbose=1,
    #     validation_data=(x_test, y_test))

    return model



#测试精度
def evaluate_first_model(model, x_test: np.ndarray, y_test: np.ndarray):
    score = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

#保存初始模型
def save_first_model(model):
    model.save(first_model_path)
    print('first_model has been saved in: ' +  first_model_path)



if __name__ == "__main__":
    train_global_model10(x_train, y_train, x_test, y_test)