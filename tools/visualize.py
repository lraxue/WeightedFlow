# -*- coding: utf-8 -*-
# @Time    : 17-10-3 上午10:06
# @Author  : Fei Xue
# @Email   : feixue@pku.edu.cn
# @File    : visualize.py
# @Software: PyCharm Community Edition
#   compute colored image to visualize optical flow file .flo

#   According to the matlab code of Deqing Sun and c++ source code of Daniel Scharstein
#   Contact: dqsun@cs.brown.edu
#   Contact: schar@middlebury.edu

#   Author: Johannes Oswald, Technical University Munich
#   Contact: johannes.oswald@tum.de
#   Date: 26/04/2017

#	For more information, check http://vision.middlebury.edu/flow/

import cv2
import sys
import numpy as np
import argparse

import tensorflow as tf


def read_flow(filename):
    with open(filename, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        assert (202021.25 == magic), 'Magic number incorrect. Invalid .flo file'
        h = np.fromfile(f, np.int32, count=1)[0]
        w = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2 * w * h)
    # Reshape data into 3D array (columns, rows, bands)
    data2D = np.resize(data, (w, h, 2))
    return data2D


def makeColorwheel():
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])  # r g b

    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY, 1) / RY)
    col += RY

    # YG
    colorwheel[col:YG + col, 0] = 255 - np.floor(255 * np.arange(0, YG, 1) / YG)
    colorwheel[col:YG + col, 1] = 255
    col += YG

    # GC
    colorwheel[col:GC + col, 1] = 255
    colorwheel[col:GC + col, 2] = np.floor(255 * np.arange(0, GC, 1) / GC)
    col += GC

    # CB
    colorwheel[col:CB + col, 1] = 255 - np.floor(255 * np.arange(0, CB, 1) / CB)
    colorwheel[col:CB + col, 2] = 255
    col += CB

    # BM
    colorwheel[col:BM + col, 2] = 255
    colorwheel[col:BM + col, 0] = np.floor(255 * np.arange(0, BM, 1) / BM)
    col += BM

    # MR
    colorwheel[col:MR + col, 2] = 255 - np.floor(255 * np.arange(0, MR, 1) / MR)
    colorwheel[col:MR + col, 0] = 255
    return colorwheel

def makeColorwheel_tf():
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = tf.zeros([ncols, 3])  # r g b

    col = 0
    # RY
    colorwheel[0:RY, 0].assign(255)
    colorwheel[0:RY, 1].assign(tf.floor(255 * tf.range(0, RY, 1) / RY))
    col += RY

    # YG
    colorwheel[col:YG + col, 0].assign(255 - tf.floor(255 * tf.range(0, YG, 1) / YG))
    colorwheel[col:YG + col, 1].assign(255)
    col += YG

    # GC
    colorwheel[col:GC + col, 1].assign(255)
    colorwheel[col:GC + col, 2].assign(tf.floor(255 * tf.range(0, GC, 1) / GC))
    col += GC

    # CB
    colorwheel[col:CB + col, 1].assign(255 - tf.floor(255 * tf.range(0, CB, 1) / CB))
    colorwheel[col:CB + col, 2].assign(255)
    col += CB

    # BM
    colorwheel[col:BM + col, 2].assign(255)
    colorwheel[col:BM + col, 0].assign(tf.floor(255 * tf.range(0, BM, 1) / BM))
    col += BM

    # MR
    colorwheel[col:MR + col, 2].assign(255 - tf.floor(255 * tf.range(0, MR, 1) / MR))
    colorwheel[col:MR + col, 0].assign(255)
    return colorwheel

def computeColor(u, v):
    colorwheel = makeColorwheel()
    nan_u = np.isnan(u)
    nan_v = np.isnan(v)
    nan_u = np.where(nan_u)
    nan_v = np.where(nan_v)

    u[nan_u] = 0
    u[nan_v] = 0
    v[nan_u] = 0
    v[nan_v] = 0

    ncols = colorwheel.shape[0]
    radius = np.sqrt(u ** 2 + v ** 2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1)  # -1~1 maped to 1~ncols
    k0 = fk.astype(np.uint8)  # 1, 2, ..., ncols
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0

    img = np.empty([k1.shape[0], k1.shape[1], 3])
    ncolors = colorwheel.shape[1]
    for i in range(ncolors):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255
        col1 = tmp[k1] / 255
        col = (1 - f) * col0 + f * col1
        idx = radius <= 1
        col[idx] = 1 - radius[idx] * (1 - col[idx])  # increase saturation with radius
        col[~idx] *= 0.75  # out of range
        img[:, :, 2 - i] = np.floor(255 * col).astype(np.uint8)

    return img.astype(np.uint8)


def computeColor_tf(u, v):
    colorwheel = makeColorwheel_tf()
    nan_u = tf.is_nan(u)
    nan_v = tf.is_nan(v)
    nan_u = tf.where(nan_u)
    nan_v = tf.where(nan_v)

    u[nan_u] = 0
    u[nan_v] = 0
    v[nan_u] = 0
    v[nan_v] = 0

    ncols = colorwheel.get_shape()[0]
    radius = tf.sqrt(u ** 2 + v ** 2)
    a = tf.atan2(-v, -u) / tf.constant(3.1415926)
    fk = (a + 1) / 2 * (ncols - 1)  # -1~1 maped to 1~ncols
    k0 = fk.astype(tf.uint8)  # 1, 2, ..., ncols
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0

    img = tf.zeros([k1.get_shape()[0], k1.get_shape()[1], 3])
    ncolors = colorwheel.get_shape()[1]
    for i in range(ncolors):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255
        col1 = tmp[k1] / 255
        col = (1 - f) * col0 + f * col1
        idx = radius <= 1
        col[idx] = 1 - radius[idx] * (1 - col[idx])  # increase saturation with radius
        col[~idx] *= 0.75  # out of range
        img[:, :, 2 - i] = tf.floor(255 * col).astype(tf.uint8)

    return img.astype(np.uint8)


def computeImg(flow):
    eps = sys.float_info.epsilon
    UNKNOWN_FLOW_THRESH = 1e9
    UNKNOWN_FLOW = 1e10

    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999
    maxv = -999

    minu = 999
    minv = 999

    maxrad = -1
    # fix unknown flow
    # greater_u = tf.where(u > UNKNOWN_FLOW_THRESH)
    # greater_v = tf.where(v > UNKNOWN_FLOW_THRESH)
    # tf.assign(u, 0, greater_u)
    # tf.assign(u, 0, greater_v)
    # tf.assign(v, 0, greater_u)
    # tf.assign(v, 0, greater_v)
    # u[greater_u] = 0
    # u[greater_v] = 0
    # v[greater_u] = 0
    # v[greater_v] = 0

    #
    # maxu = tf.maximum(maxu, tf.maximum(u))
    # minu = tf.minimum(minu, tf.minimum(u))
    #
    # maxv = tf.maximum(maxv, tf.maximum(v))
    # minv = tf.minimum(minv, tf.minimum(v))
    rad = tf.sqrt(tf.multiply(u, u) + tf.multiply(v, v))
    maxrad = tf.maximum(tf.cast(maxrad, tf.float32), tf.reduce_max(rad))
    # print('max flow: %.4f flow range: u = %.3f .. %.3f; v = %.3f .. %.3f\n' % (maxrad, minu, maxu, minv, maxv))

    u = u / (maxrad + eps)
    v = v / (maxrad + eps)
    img = computeColor_tf(u, v)
    return img




# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         '--flowfile',
#         type=str,
#         default='colorTest.flo',
#         help='Flow file'
#     )
#     parser.add_argument(
#         '--write',
#         type=bool,
#         default=False,
#         help='write flow as png'
#     )
#     file = parser.parse_args().flowfile
#     flow = read_flow(file)
#     img = computeImg(flow)
#     # cv2.imshow('Flow Image',img)
#     # k = cv2.waitKey()
#     if parser.parse_args().write:
#         cv2.imwrite(file[:-4] + '.png', img)
