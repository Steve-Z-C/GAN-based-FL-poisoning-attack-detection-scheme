from __future__ import print_function, division
import fed_learn
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import datasets, layers, models, optimizers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Reshape, LeakyReLU
from tensorflow.keras.layers import  UpSampling2D, Conv2D, MaxPooling2D, BatchNormalization, Activation, ZeroPadding2D
from tensorflow.compat.v1 import ConfigProto
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt 
import numpy as np

config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


class GAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels) #输入/生成的图片尺寸
        self.latent_dim = 784
 
        optimizer = Adam(0.0001, 0.5)
 
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
 
        # Build the generator
        self.generator = self.build_generator()
 
        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)
 
        # For the combined model we will only train the generator
        self.discriminator.trainable = False
 
        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)
 
        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
 

    def build_generator(self):
 
        # model = Sequential()
 
        # model.add(Dense(256, input_dim=self.latent_dim))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Dense(512))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Dense(1024))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        # model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        # model.add(Reshape(self.img_shape))
        
        model = Sequential()
        model.add(Dense(input_dim=784, units=1024))
        model.add(Activation('tanh'))
        model.add(Dense(128*7*7))
        # model.add(Dense(units=28*28*1))
        model.add(BatchNormalization())
        model.add(Activation('tanh'))
        model.add(Reshape((7, 7, 128), input_shape=(128*7*7,),))
        # model.add(Reshape((28, 28, 1), input_shape=(28*28*1,),))
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Conv2D(64, (5, 5), padding='same'))
        model.add(Activation('tanh'))
        model.add(UpSampling2D(size=(2, 2)))
        model.add(Conv2D(1, (5, 5), padding='same'))
        model.add(Activation('tanh'))
        model.summary()
 
        noise = Input(shape=(self.latent_dim,))
        img = model(noise)
 
        return Model(noise, img)
 
    def build_discriminator(self):
        model = Sequential()
        model.add(
                Conv2D(64, (5, 5),
                padding='same',
                input_shape=(28, 28, 1))
                )
        model.add(Activation('tanh'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, (5, 5)))
        model.add(Activation('tanh'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation('tanh'))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        return model
        model = load_model("./models/global model/global_model.h5")
        model.pop()
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
 
        img = Input(shape=self.img_shape)
        validity = model(img)
 
        return Model(img, validity)
 
    def train(self, epochs, batch_size, sample_interval):
 
        # Load the dataset
        # x_train, y_train = fed_learn.load_mnist('/dataset/Basic_data', kind='train')
        x_test, Y_train = fed_learn.load_mnist('/dataset/Basic_data', kind='t10k')

        # (_, _), (x_test, Y_train) = tf.keras.datasets.mnist.load_data("/mnist.npz")

        # Rescale -1 to 1
        # X_train = X_train / 127.5 - 1.
        X_train = x_test / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)
 
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
 
        j = 0
        x_train_0 = {}
        x_noise = {}
        for i in range(len(X_train)):
            if j < batch_size:
                if Y_train[i]==1:
                    x_train_0[j] = X_train[i]
                    j += 1
        x_train_0 = list(x_train_0.values())

        # # x_train = np.array(x_train)
        # for i in range(len(x_train)):
        #     m = x_train[i].flatten()
        #     noise_1 = np.random.normal(0, 4, len(m))
        #     noise_2 = np.random.poisson(lam=4, size=None)
        #     m = m + noise_1 + noise_2
        #     noise_3 = np.random.normal(0, 15, 580)
        #     m[120:700] += noise_3
        #     x_noise[i] = np.array(m).reshape(784,)
        # x_noise = list(x_noise.values())
        # x_noise = np.array(x_noise)

        gan_test = {}
        for epoch in range(epochs):
 
            # ---------------------
            #  Train Discriminator
            # ---------------------
 
            # Select a random batch of images
            # idx = np.random.randint(0, X_train.shape[0], batch_size)
            # imgs = X_train[idx]
            x_real = {}
            for i in range(len(x_train_0)):
                x_real[i] = np.array(x_train_0[i]).reshape(28,28,1)
            x_real = list(x_real.values())
            x_real = np.array(x_real)

            imgs = x_real
 
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new images
            # x = x_noise[epoch%len(x_noise)].flatten()
            gen_imgs = self.generator.predict(noise)
 
            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
 
            # ---------------------
            #  Train Generator
            # ---------------------
 
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
 
            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)
            # g_loss = self.combined.train_on_batch(x_noise, valid)
            
 
            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))


            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch/sample_interval)
                noise = np.random.normal(0, 1, (1, self.latent_dim))
                gen_imgs = self.generator.predict(noise)
                gan_test[(epoch/sample_interval)] = gen_imgs
        gan_test = list(gan_test.values())
        gan_test = np.array(gan_test).reshape(len(gan_test), 28, 28, 1)
        np.save("./generator sample/picture 1/gan_test_1.npy", gan_test)
 
    def sample_images(self, epoch):
        # r, c = 5, 5
        r, c = 1, 1
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        # gen_imgs = self.generator.predict(x_noise[epoch%256].flatten())
        gen_imgs = self.generator.predict(noise)
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        # fig, axs = plt.subplots(r, c)

        # for i in range(r):
        #     for j in range(c):
        #         axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
        #         axs[i,j].axis('off')
        #         cnt += 1
        plt.imshow(gen_imgs[0, :,:,0], cmap='gray')
        # plt.axis('off')
        # cnt += 1
        plt.savefig(r"/home/whb/Desktop/poison_attack/fashion_mnist/generator sample/picture 1/%d.png" % epoch)
        plt.close()
 
 
if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=150000, batch_size=512, sample_interval=15)
    # epoch = 500
    # for i in range(epoch):
    #     r, c = 1, 1
    #     noise = np.random.normal(0, 1, (r * c, 100))
    #     gen_imgs = gan.generator.predict(noise)
    #     gen_imgs = 0.5 * gen_imgs + 0.5
    #     fig, axs = plt.subplots(r, c)
    #     cnt = 0
    #     # for i in range(r):
    #     #     for j in range(c):
    #     #         axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
    #     #         axs[i,j].axis('off')
    #     #         cnt += 1
    #     axs.imshow(gen_imgs[cnt, :,:,0], cmap='gray')
    #     axs.axis('off')
    #     # cnt += 1
    #     fig.savefig(r"C:\Users\Lenovo\Desktop\gan_picture/%d.png" % i)
    #     plt.close()