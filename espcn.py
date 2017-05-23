import tensorflow as tf
import numpy as np
import os
import sys

class ESPCN():

    def __init__(self, train=True):
        self.train=train

    def create_network(self, input_images):
        # input_images = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])

        Kernel1 = tf.get_variable(name='kerner1', shape=[5, 5, 3, 6],
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        Bias1 = tf.get_variable(name='Bias1', shape=[6], initializer=tf.contrib.layers.xavier_initializer())
        Conv1 = tf.nn.conv2d(input_images, Kernel1, strides=[1, 1, 1, 1], padding='VALID') + Bias1
        Activation1 = tf.nn.tanh(Conv1)

        Kernel2 = tf.get_variable(name='kerner2', shape=[3, 3, 6, 12],
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        Bias2 = tf.get_variable(name='Bias2', shape=[12], initializer=tf.contrib.layers.xavier_initializer())
        Conv2 = tf.nn.conv2d(Activation1, Kernel2, strides=[1, 1, 1, 1], padding='VALID') + Bias2
        Activation2 = tf.nn.tanh(Conv2)

        Kernel3 = tf.get_variable(name='kerner3', shape=[3, 3, 12, 24],
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        Bias3 = tf.get_variable(name='Bias3', shape=[24], initializer=tf.contrib.layers.xavier_initializer())
        Conv3 = tf.nn.conv2d(Activation2, Kernel3, strides=[1, 1, 1, 1], padding='VALID') + Bias3
        Activation3 = tf.nn.tanh(Conv3)

        Kernel4 = tf.get_variable(name='kerner4', shape=[3, 3, 24, 48],
                                  initializer=tf.contrib.layers.xavier_initializer_conv2d())
        Bias4 = tf.get_variable(name='Bias4', shape=[48], initializer=tf.contrib.layers.xavier_initializer())
        Conv4 = tf.nn.conv2d(Activation3, Kernel4, strides=[1, 1, 1, 1], padding='VALID') + Bias4

        return Conv4

    def img_resize(self, input_data):
        input_data = np.multiply(input_data, 1.0 / 255.0)
        return input_data

    def loss(self, output, lables):
        residual = output - lables
        loss = tf.square(residual)
        reduced_loss = tf.reduce_mean(loss)
        tf.summary.scalar('loss', reduced_loss)
        return reduced_loss

    def save(self, sess, saver, logdir, step):
        # print('[*] Storing checkpoint to {} ...'.format(logdir), end="")
        sys.stdout.flush()

        if not os.path.exists(logdir):
            os.makedirs(logdir)

        checkpoint = os.path.join(logdir, "model.ckpt")
        saver.save(sess, checkpoint, global_step=step)
        # print('[*] Done saving checkpoint.')

    def load(self, sess, saver, logdir):
        print("[*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(logdir)

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(logdir, ckpt_name))
            return True
        else:
            return False

    def make_image(self, lr_image):
        lr_image = self.preprocess([lr_image, None])[0]
        sr_image = self.create_network(lr_image)
        sr_image = sr_image * 255.0
        sr_image = tf.cast(sr_image, tf.int32)
        sr_image = tf.maximum(sr_image, 0)
        sr_image = tf.minimum(sr_image, 255)
        sr_image = tf.cast(sr_image, tf.uint8)
        return sr_image


if __name__ == "__main__":
    model = ESPCN()
