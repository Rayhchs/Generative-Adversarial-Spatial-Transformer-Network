"""
Created on Sun Jan 09 2022

network

@author: Ray
"""
from tf_unit import *
from utils import *
import tensorflow.compat.v1 as tf
import numpy as np
import random, time
from PIL import Image


class network():

    def __init__(self, sess, params):
        self.sess = sess
        self.k_initilizer = tf.random_normal_initializer(0, 0.02)
        self.ker_init = tf.constant_initializer([0, 0, 0, 0, 0, 0])
        self.bias_init = tf.constant_initializer([1., 0, 0, 0, 1., 0])
        self.input_size = params['input_size']
        self.batch_size = params['batch_size']
        self.epochs = params['epochs']
        self.learning_rate = params['learning_rate']
        self.beta1 = params['beta1']
        self.beta2 = params['beta2']
        self.save_epoch = params['save_epoch']
        self.bulid_model()


    # Generator
    def generator(self, x, reuse=None, name="generator"):

        with tf.variable_scope(name, reuse=reuse):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False

            h = LocalNet(x, self.k_initilizer, reuse=reuse, name='STN1')
            h = tf.layers.dense(h, 6, kernel_initializer=self.ker_init, bias_initializer=self.bias_init)

            htrans = transformer(x, h)

            return htrans, h


    # PatchGAN Discriminator
    def discriminator(self, y, reuse=None, name="discriminator"):

        with tf.variable_scope(name, reuse=reuse):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False

            h = d_block(y, 64, self.k_initilizer,kernel_sz=3, strides=1, reuse=reuse, do_norm=False, name='d64_1')
            h = d_block(h, 128, self.k_initilizer, strides=2, reuse=reuse, do_norm=True, name='d128_1')
            h = d_block(h, 256, self.k_initilizer, strides=2, reuse=reuse, do_norm=True, name='d256_1')
            h = d_block(h, 512, self.k_initilizer, strides=1, reuse=reuse, do_norm=True, name='d512_1')
            h = d_block(h, 1, self.k_initilizer, strides=1, reuse=reuse, do_norm=False, act='sigmoid', name='output')

            return h


    # Define Loss
    def loss(self, real, fake, y, g, Lambda=1):

        # Discriminator Loss: Vanillna Loss
        loss_d_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real, labels=tf.ones_like(real)))
        loss_d_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake, labels=tf.zeros_like(fake)))
        loss_d = loss_d_real + loss_d_fake

        # GAN Loss: Vanillna Loss + L2 loss
        loss_g_gan = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake, labels=tf.ones_like(real)))
        mse = tf.keras.losses.MeanSquaredError()
        l2_loss = mse(y, g)

        loss_g = loss_g_gan + l2_loss*Lambda
        return loss_g, loss_d, l2_loss


    # Build model
    def bulid_model(self):

        # init variable
        self.x_ = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.input_size, self.input_size, 3], name='x')
        self.y_ = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.input_size, self.input_size, 3], name='y')

        # generator
        self.g_, self.params = self.generator(self.x_)
        
        # discriminator
        self.real = self.discriminator(self.y_)
        self.fake = self.discriminator(self.g_, reuse=True)

        # loss
        self.loss_g, self.loss_d, self.l2_loss = self.loss(self.real, self.fake, self.y_, self.g_)

        # summary
        tf.summary.image("Input image", self.x_)
        tf.summary.image("Final image", self.g_)
        tf.summary.image("Ground truth", self.y_)

        tf.summary.scalar("t1", tf.reduce_mean(self.params[:,0]))
        tf.summary.scalar("t2", tf.reduce_mean(self.params[:,1]))
        tf.summary.scalar("t3", tf.reduce_mean(self.params[:,2]))
        tf.summary.scalar("t4", tf.reduce_mean(self.params[:,3]))
        tf.summary.scalar("t5", tf.reduce_mean(self.params[:,4]))
        tf.summary.scalar("t6", tf.reduce_mean(self.params[:,5]))   
        tf.summary.scalar("Generator loss", self.loss_g)
        tf.summary.scalar("Discriminator loss", self.loss_d)
        tf.summary.scalar("L2_loss", self.l2_loss)
        self.merged = tf.summary.merge_all()
        
        # vars
        self.vars_g = [var for var in tf.trainable_variables() if var.name.startswith('generator')]
        self.vars_d = [var for var in tf.trainable_variables() if var.name.startswith('discriminator')]

        # saver
        self.saver = tf.train.Saver()


    # Train
    def train(self, image_lists):
      
        # Optimizer
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_step_d = tf.train.AdamOptimizer(learning_rate=self.learning_rate, 
                                                beta1=self.beta1, beta2=self.beta2).minimize(self.loss_d, var_list=self.vars_d)
        train_step_g = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                beta1=self.beta1, beta2=self.beta2).minimize(self.loss_g, var_list=(self.vars_g))

        # init variable
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter("./log", self.sess.graph)
        #self.saver.restore(self.sess, tf.train.latest_checkpoint('./checkpoint'))
        
        # Training
        for i in range(self.epochs):

            decay = 1000
            if i >= decay and i < 2*decay:
                train_step_d.learning_rate = self.learning_rate - (0.9*self.learning_rate*(i-decay)/(decay))

            else:
                train_step_d.learning_rate = self.learning_rate*0.1

            batch_num = int(np.ceil(len(image_lists) / self.batch_size))
            batch_list = np.array_split(image_lists, batch_num)
            random.shuffle(batch_list)
            t1 = time.time()

            loss_ds = []
            loss_gs = []
            for j in range(len(image_lists)):

                batch_x, batch_y = load_data(batch_list[j], self.input_size)
                _, loss_d = self.sess.run([train_step_d, self.loss_d],
                                          feed_dict={self.x_: batch_x, self.y_: batch_y})

                _, loss_g = self.sess.run([train_step_g, self.loss_g],
                                            feed_dict={self.x_: batch_x, self.y_: batch_y})
                loss_ds.append(loss_d)
                loss_gs.append(loss_g)   
                            
            loss_ds = sum(loss_ds) / len(image_lists)
            loss_gs = sum(loss_gs) / len(image_lists)
            
            print("Epoch: %d/%d: discriminator loss: %.4f; generator loss: %.4f; training time: %.1fs" % ((i + 1), self.epochs, loss_ds, loss_gs, time.time()-t1))

            # Save loss
            summary = self.sess.run(self.merged, feed_dict={self.x_: batch_x, self.y_: batch_y})
            self.writer.add_summary(summary, global_step=i)
            
            # Save model
            if (i + 1) % self.save_epoch == 0:
                self.saver.save(self.sess, './checkpoint/epoch_%d.ckpt' % (i + 1))
            

    # Test
    def test(self, paths):

        # init variables
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        # load model
        self.saver.restore(self.sess, tf.train.latest_checkpoint('./checkpoint'))

        im = []
        for j in range(len(paths)):
            batch_x = load_test_data(paths[j], self.input_size)

            g = self.sess.run(self.g_, feed_dict={self.x_: batch_x})

            g = (np.array(g[0]) + 1) * 127.5
            im.append(Image.fromarray(np.uint8(g)))

        return im