# -*- coding: utf-8 -*-
# @Time    : 17-10-4 下午4:10
# @Author  : Fei Xue
# @Email   : feixue@pku.edu.cn
# @File    : test_transform.py
# @Software: PyCharm Community Edition

import os
import tensorflow as tf
from bilinear_sampler import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# I = tf.constant(1, tf.float32, [1, 4, 4, 1])
I = tf.random_uniform([1, 3, 4, 1], 0, 1, tf.float32)
flow = tf.random_normal([1, 3, 4, 2], 0, 0, tf.float32)

I_bil = tf.image.resize_bilinear(I, [3, 4])

I_est_h = bilinear_sampler_1d_h(I, flow[:,:,:,0])
x_t, y_t, I_est = bilinear_sampler(I, flow)


sess = tf.Session()

print(sess.run(I))
print('\n')

print(sess.run(flow))
print('\n')

print(sess.run(I_est))
print('\n')

print(sess.run(I_bil))
print('\n')

print(sess.run(I_est_h))
print('\n')

print(sess.run(x_t))
print(sess.run(y_t))
print('\n')