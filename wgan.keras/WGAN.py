# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from keras.constraints import Constraint
from tqdm import tqdm
from utils import *
from sampler import *

class WeightClipConstraint(Constraint):
    '''Clips the weights incident to each hidden unit to be inside a range
    '''
    def __init__(self, clip_value):
        self.clip_value = clip_value

    def __call__(self, weights):
        return keras.backend.clip(weights, -self.clip_value, self.clip_value)

    def get_config(self):
        return {'clip_value': self.clip_value}

class WassersteinGAN(object):
    def __init__(self, x_sampler, z_sampler, z_dim, logger):
        self.logger = logger
        self.generator = self.Generator(z_dim)
        self.discriminator = self.Discriminator()
        self.gan = self.GAN(self.generator, self.discriminator)
        self.D_loss = []
        self.G_loss = []
        self.x_sampler = x_sampler
        self.z_sampler = z_sampler
    
    def wasserstein_loss(self, y_true, y_pred):
        return keras.backend.mean(y_true * y_pred)

    def Generator(self, z_dim):
        init = keras.initializers.RandomNormal(stddev=0.01)
        model = keras.models.Sequential([
                            keras.layers.Dense(4 * 4 * 512, input_shape=[z_dim]),
                            keras.layers.Reshape([4, 4, 512]),
                            keras.layers.BatchNormalization(momentum=0.5),
                            keras.layers.ReLU(),
                            keras.layers.Conv2DTranspose(filters=256, kernel_size=(5, 5), strides=(2, 2), padding='same', kernel_initializer=init),
                            keras.layers.BatchNormalization(momentum=0.5),
                            keras.layers.ReLU(),
                            keras.layers.Conv2DTranspose(filters=128, kernel_size=(5, 5), strides=(2, 2), padding='same', kernel_initializer=init),
                            keras.layers.BatchNormalization(momentum=0.5),
                            keras.layers.ReLU(),
                            keras.layers.Conv2DTranspose(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', kernel_initializer=init),
                            keras.layers.BatchNormalization(momentum=0.5),
                            keras.layers.ReLU(),
                            keras.layers.Conv2DTranspose(filters=3, kernel_size=(5, 5), strides=(2, 2), padding='same', kernel_initializer=init, activation='tanh')
        ])
        
        optimizer = keras.optimizers.RMSprop(lr=0.00005)
        model.compile(loss=self.wasserstein_loss, optimizer=optimizer)
        self.logger.info(model.summary())
        return model

    def Discriminator(self, input_shape=(64, 64, 3)):
        init = keras.initializers.RandomNormal(stddev=0.01)
        constraint = WeightClipConstraint(0.01)
        model = keras.models.Sequential([
                                keras.layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', input_shape=input_shape, kernel_initializer=init, kernel_constraint=constraint),
                                keras.layers.LeakyReLU(0.2),
                                keras.layers.Conv2D(filters=128, kernel_size=(5, 5), strides=(2, 2), padding='same', kernel_initializer=init, kernel_constraint=constraint),
                                keras.layers.BatchNormalization(momentum=0.5),
                                keras.layers.LeakyReLU(0.2),
                                keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(2, 2), padding='same', kernel_initializer=init, kernel_constraint=constraint),
                                keras.layers.BatchNormalization(momentum=0.5),
                                keras.layers.LeakyReLU(0.2),
                                keras.layers.Conv2D(filters=512, kernel_size=(5, 5), strides=(1, 1), padding='same', kernel_initializer=init, kernel_constraint=constraint),
                                keras.layers.BatchNormalization(momentum=0.5),
                                keras.layers.LeakyReLU(0.2),
                                keras.layers.Flatten(),
                                keras.layers.Dense(1, activation='linear')
        ])
        optimizer = keras.optimizers.RMSprop(lr=0.00005)
        model.compile(loss=self.wasserstein_loss, optimizer=optimizer)
        self.logger.info(model.summary())
        return model

    def GAN(self, generator, discriminator):
        discriminator.trainable = False
        model = keras.models.Sequential([generator, discriminator])
        optimizer = keras.optimizers.RMSprop(lr=0.00005)
        model.compile(loss=self.wasserstein_loss, optimizer=optimizer) 
        return model

    def sampleDataImage(self):
        fig, a = plt.subplots(5, 5, figsize=(10, 10))
        for i in range(5):
          for j in range(5):
            a[i][j].imshow(self.x_sampler.sample(1))
            a[i][j].set_xticks([])
            a[i][j].set_yticks([])
        plt.show()


    def train(self, batch_size, seed, n_steps, n_discriminator=5):
        for step in range(n_steps):
            self.logger.info("Step: " + str(step))
            for _ in range(n_discriminator):
                X_batch = self.x_sampler.sample(batch_size)
                noise = self.z_sampler.sample(batch_size)
                generated_images = self.generator(noise)
                X_all = tf.concat([generated_images, X_batch], axis=0)
                y1 = tf.constant([[-1.]] * batch_size + [[1.]] * len(X_batch))
                self.discriminator.trainable = True
                self.D_loss.append(self.discriminator.train_on_batch(X_all, y1))
    
            noise = self.z_sampler.sample(2 * batch_size)
            y2 = tf.constant([[1.]] * (batch_size * 2))
            self.discriminator.trainable = False
            self.G_loss.append(self.gan.train_on_batch(noise, y2))
            if(step%50 == 0):
                display.clear_output(wait=True)
                if(step%100 == 0):
                    generate_and_save_images(self.generator, step+1, seed)
                    self.logger.info("Saved weight at step: " + str(step))
                    self.gan.save_weights('model_weight')
                else:
                    generate_and_save_images(self.generator, step+1, seed, False)

    def load_model(self, filename):
        self.logger.info("Model loaded...")
        self.gan.load_weights(filename)
