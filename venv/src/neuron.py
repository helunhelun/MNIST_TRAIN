# coding:utf-8

import os
import re
import cPickle
import tensorflow as tf
import sys
import cv2 as cv
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


CIFAR_DATA_DIR = '../data/cifar-10-batches-py'


class Neuron:
    def __init__(self):
        self.x = tf.placeholder(tf.float32,[None,3072])
        #self.w = tf.get_variable('w',[self.x.get_shape()[-1],1],initializer=tf.random_normal_initializer(0,1))
        self.w = tf.Variable(tf.zeros([3072,10]))
        self.b = tf.Variable(tf.ones([10]))
        self.y = tf.nn.softmax(tf.matmul(self.x,self.w)+self.b)
        self.y_ = tf.placeholder(tf.float32,[None,10])

    def train_net(self,data,label):
        '''神经网络训练 输入为图片数据，标签'''
        loss = tf.reduce_mean(tf.square(self.y - self.y_))
        cross_entropy = -tf.reduce_sum(self.y_ * tf.log(self.y))
        #利用反向梯度算法 要计算的模型
        train_net = tf.train.AdamOptimizer(1e-3).minimize(loss)
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        sess.run(train_net,feed_dict={self.x:data,self.y_:label})
        return sess

    def assess_net(self,sess,data,labels):
        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        result = sess.run(accuracy, feed_dict={self.x: data, self.y_: labels})
        print result
        return  result

class CifarData:
    def __init__(self,filenames,need_shuffle):
        all_data = []
        all_label = []
        for filename in filenames:
            data,labels = self._load_data(filename)
            all_data.append(data)
            all_label.append(labels)
        self._data = np.vstack(all_data) # 将all_data列表类型变成np.narray类型
        self._data = self._data / 127.5 - 1
        self._labels = np.hstack(all_label) # 将all_label列表变成np.narray类型
        self._num_case = self._data.shape[0]
        print self._num_case
        self._need_shuffle = need_shuffle
        self._indicator = 0
        if self._need_shuffle :
            self._shuffle_data()


    def _load_data(self,filename):
        '''加载数据,输入文件名，输出文件数据以及标签'''
        with open(filename, 'rb') as f:
            data = cPickle.load(f)
            return data['data'], data['labels']

    def _shuffle_data(self):
        '''乱序排列'''
        p = np.random.permutation(self._num_case)   # 生成一个乱序的数组
        self._data = self._data[p]   # 乱序排列
        self._labels = self._labels[p] # 乱序排列

    def next_batch(self,batch_size):
        '''返回一批数据 输入批数据'''
        end_indicator = self._indicator + batch_size
        if end_indicator > self._num_case:
            if self._need_shuffle   :
                self._shuffle_data()
                self._indicator = 0
                end_indicator = batch_size
            else:
                raise Exception('have no more example')
        if end_indicator > self._num_case:
            raise Exception('batch is more than all example data')
        batch_data = self._data[self._indicator : end_indicator]
        batch_labels = self._labels[self._indicator : end_indicator]
        batch_labels = tf.one_hot(batch_labels,10,dtype=tf.float32)
        sessto = tf.Session()
        sessto.run(tf.global_variables_initializer())
        batch_labels = sessto.run(batch_labels)
        self._indicator = end_indicator #一批完成之后 开始索引更新
        return batch_data , batch_labels





if __name__ == '__main__':
        print 'mian is runing'
        print os.listdir(CIFAR_DATA_DIR)
        train_data = [os.path.join(CIFAR_DATA_DIR,'data_batch_%d' %i) for i in range(1,5)]
        test_data = [os.path.join(CIFAR_DATA_DIR,'test_batch')]
        cifardata = CifarData(train_data,True)
        batch_xs ,batch_ys = cifardata.next_batch(40000)
        print batch_xs
        print batch_ys
        print type(batch_ys)
        neuron = Neuron()
        sess = neuron.train_net(batch_xs,batch_ys)
        cifardatatest = CifarData(test_data,False)
        batch_xs,batch_ys = cifardatatest.next_batch(10000)
        print neuron.assess_net(sess,batch_xs,batch_ys)

