# coding:utf-8

import os
import re
import tensorflow as tf
import sys
import cv2 as cv
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

MNIST_DATA_DIR = '../data/MNIST_DATABASE'

class Neural:
    def __init__(self):
        self.x = tf.placeholder(tf.float32 , [None,784] , 'x')
        self.y_ = tf.placeholder(tf.float32, [None, 10])
        self._x_image = tf.reshape(self.x,[-1,28,28,1]) # 将 输入的图像变成4d，分别对应[batch,height,width,channel]
        self._w_conv1 = self.weight_variable([5,5,1,32])  # 第一层的卷积核
        self._b_conv1 = self.bias_variable([32])
        #把x_image和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max pooling。
        self._h_conv1 = tf.nn.relu(self.conv2d(self._x_image,self._w_conv1)+self._b_conv1)
        self._h_pool1 = self.max_pool_2x2(self._h_conv1)
        # 第二层卷积
        self._w_conv2 = self.weight_variable([5,5,32,64])
        self._b_conv2 = self.bias_variable([64])
        self._h_conv2 = tf.nn.relu(self.conv2d(self._h_pool1,self._w_conv2)+self._b_conv2)
        self._h_pool2 = self.max_pool_2x2(self._h_conv2)
        # 密集连接层
        self._w_fc1 = self.weight_variable([7*7*64,1024])
        self._b_fc1 = self.bias_variable([1024])

        self._h_pool2_flat = tf.reshape(self._h_pool2,[-1,7*7*64])
        self._h_fc1 = tf.nn.relu(tf.matmul(self._h_pool2_flat,self._w_fc1)+self._b_fc1)
        # Dropout
        self._keep_prob = tf.placeholder("float")
        self._h_fc1_drop = tf.nn.dropout(self._h_fc1,self._keep_prob)
        # 输出层
        self._w_fc2 = self.weight_variable([1024,10])
        self._b_fc2 = self.bias_variable([10])
        #输出层激活函数
        self._y_conv = tf.nn.softmax(tf.matmul(self._h_fc1_drop,self._w_fc2)+self._b_fc2)
    def load_data(self):
        mnist = input_data.read_data_sets(MNIST_DATA_DIR,one_hot=True)
        print mnist.train.images.shape
        print mnist.test.labels.shape
        return mnist


    def train_net(self,mnist):
        cross_entropy = -tf.reduce_sum(self.y_*tf.log(self._y_conv))
        #train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)
        train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
        #train_step = tf.train.AdagradOptimizer(1e-4).minimize(cross_entropy)
        init = tf.global_variables_initializer()
        sess = tf.InteractiveSession()
        sess.run(init)

        for i in range(550):
            batch_xs,batch_ys = mnist.train.next_batch(100)
            sess.run(train_step,feed_dict={self.x:batch_xs,self.y_:batch_ys, self._keep_prob: 0.5})
        return sess

    def assess_net(self,sess,mnist):
        correct_prediction = tf.equal(tf.arg_max(self._y_conv,1),tf.arg_max(self.y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))
        return sess.run(accuracy,feed_dict = {self.x:mnist.test.images,self.y_:mnist.test.labels,self._keep_prob: 1.0})

    def conv2d(self,x,w):
        '''卷积层'''
        # x 为输入源图像 [batch,height,width,channel]  w为卷积核 [filter_height,filter_width,in_channel,out_channel]
        return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')   # strides 为步长 padding为间距

    def max_pool_2x2(self,x):
        '''池化层'''
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    def weight_variable(self,shape):
        '''产生一个标准差为0.1的正态分布的shape'''
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        '''将偏置量b常量为0.1'''
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

if __name__ == '__main__':
    print os.listdir(MNIST_DATA_DIR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    neural = Neural()
    mnist = neural.load_data()
    sess  = neural.train_net(mnist)
    result = neural.assess_net(sess,mnist)
    print '准确率为:',result



