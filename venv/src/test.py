# coding:utf-8

import os
import sys
import tensorflow as tf
import numpy as np
import cv2 as cv


if __name__ == '__main__':
    #p = np.random.permutation(100)
   # print p
    c = tf.truncated_normal(shape=[10,10],mean=0,stddev=0.1)
    v =  tf.Variable(c)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print sess.run(v)



