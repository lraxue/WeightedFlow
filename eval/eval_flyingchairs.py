# -*- coding: utf-8 -*-
# @Time    : 17-10-6 下午3:59
# @Author  : Fei Xue
# @Email   : feixue@pku.edu.cn
# @File    : eval_flyingchairs.py
# @Software: PyCharm Community Edition


import cv2
import numpy as np
import os
import tensorflow as tf
import sys
import argparse

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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
    greater_u = np.where(u > UNKNOWN_FLOW_THRESH)
    greater_v = np.where(v > UNKNOWN_FLOW_THRESH)
    u[greater_u] = 0
    u[greater_v] = 0
    v[greater_u] = 0
    v[greater_v] = 0

    maxu = max([maxu, np.amax(u)])
    minu = min([minu, np.amin(u)])

    maxv = max([maxv, np.amax(v)])
    minv = min([minv, np.amin(v)])
    rad = np.sqrt(np.multiply(u, u) + np.multiply(v, v))
    maxrad = max([maxrad, np.amax(rad)])
    # print('max flow: %.4f flow range: u = %.3f .. %.3f; v = %.3f .. %.3f\n' % (maxrad, minu, maxu, minv, maxv))

    u = u / (maxrad + eps)
    v = v / (maxrad + eps)
    img = computeColor(u, v)
    return img


def epe(input_flow, target_flow):
    return tf.norm(target_flow - input_flow, 2, 1)


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


def load_flow_from_file(filepath):
    flow = np.load(filepath)
    return flow


def visualzie_flow(flow, save_flow=False, save_dir=None, file=None):
    img = computeImg(flow)
    cv2.imshow('Flow Image', img)
    k = cv2.waitKey(100)
    if save_flow:
        cv2.imwrite(os.path.join(save_dir, file[:-4] + '.png'), img)


def visualize_flow_from_file(dataset_dir, file, save_flow=True, save_dir=None):
    flow = read_flow(os.path.join(dataset_dir, file))
    img = computeImg(flow)
    cv2.imshow('Flow Image', img)
    k = cv2.waitKey(10)
    if save_flow:
        cv2.imwrite(os.path.join(save_dir, file[:-4] + '.png'), img)


def var_mean(flow_to_mean):
    """Pyfunc wrapper for the confidence / mean calculation"""

    def _var_mean(flow_to_mean):
        """ confidence / mean calculation"""
        flow_to_mean = np.array(flow_to_mean)
        x = flow_to_mean[:, :, :, 0]
        y = flow_to_mean[:, :, :, 1]
        var_x = np.var(x, 0)
        var_y = np.var(y, 0)
        # var_mea = np.mean(np.array([var_x, var_y]), 0)
        var_mea = (var_x + var_y) / 2
        # TODO check /2, /8 here ...
        var_mea = np.exp(-1 * np.array(var_mea, np.float32) / 8)
        # normalize
        flow_x_m = np.mean(x, 0)
        flow_y_m = np.mean(y, 0)
        flow_to_mean = np.zeros(list([384, 512, 3]), np.float32)
        flow_to_mean[:, :, 0] = flow_x_m
        flow_to_mean[:, :, 1] = flow_y_m
        var_img = np.zeros(list([384, 512, 3]), np.float32)
        var_img[:, :, 0] = var_mea
        var_img[:, :, 1] = var_mea
        var_img[:, :, 2] = var_mea
        return [flow_to_mean, var_mea, var_img]

    solved_data = tf.py_func(_var_mean, [flow_to_mean], [tf.float32, tf.float32, tf.float32], name='flow_mean')
    mean, var, var_img = solved_data[:]
    mean = tf.squeeze(tf.stack(mean))
    var = tf.squeeze(tf.stack(var))
    var_img = tf.squeeze(tf.stack(var_img))
    mean.set_shape(list([384, 512, 3]))
    var.set_shape(list([384, 512, 3][:2]) + [1])
    var_img.set_shape(list([384, 512, 3]))
    return mean, var, var_img


def slice_vector(vec, size):
    x = tf.slice(vec, [0, 0, 0, 0], [size] + [384, 512, 2][:2] + [1])
    y = tf.slice(vec, [0, 0, 0, 1], [size] + [384, 512, 2][:2] + [1])
    return tf.squeeze(x), tf.squeeze(y)


def epe(input_flow, target_flow):
    h, w = input_flow.shape[0], input_flow.shape[1]
    flow_u = np.reshape(target_flow[:, :, 0] - input_flow[:, :, 0], [h * w])
    flow_v = np.reshape(target_flow[:, :, 1] - input_flow[:, :, 1], [h * w])

    error = 0
    for i in range(w * h):
        e = np.sqrt(flow_u[i] * flow_u[i] + flow_v[i] * flow_v[i])
        error += e

    return error / (w * h)

def aee_f(gt, calc_flows, size=1):
    "average end point error"
    square = tf.square(gt - calc_flows)
    x, y = slice_vector(square, size)
    sqr = tf.sqrt(tf.add(x, y))
    aee = tf.metrics.mean(sqr)
    return aee


def compute_error(flow, calc_flows):
    # Get flow tensor from flownet model


    flow_mean, confidence, conf_img = var_mean(calc_flows)

    # confidence = tf.image.convert_image_dtype(confidence, tf.uint16)
    # calc EPE / AEE = ((x1-x2)^2 + (y1-y2)^2)^1/2
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3478865/

    aee = aee_f(flow, flow_mean, 1)
    # bilateral solverc
    # img_0 = tf.squeeze(tf.stack(img_0))
    # flow_s = tf.squeeze(tf.stack(flow))
    # solved_flow = flownet.bil_solv_var(img_0, flow_mean, confidence, flow_s)
    # aee_bs = aee_f(flow, solved_flow, var_num)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--flowfile', type=str, default='../log/my_model/flows.npy', help='Flow file')
    parser.add_argument('--save_dir', type=str, default='../result/', help='save dir')
    parser.add_argument('--testfilelist', type=str, default='colorTest.flo', help='test file list')
    parser.add_argument('--dataset_dir', type=str, default='/home/fei/Data/fei/flow/FlyingChairs_release/data/',
                        help='dataset dir')

    with open('../flyingchairs_test.txt') as f:
        files = f.readlines()

    flow_gt = []
    # for i in range(len(files)):
    #     line = files[i].strip().split()[2]   # flow file
    #     flow_gt.append(os.path.join(parser.parse_args().dataset_dir, line))
    # visualize_flow_from_file(dataset_dir='/home/fei/Data/fei/flow/FlyingChairs_release/data/', file=line, save_dir='/home/fei/Research/Code/DeepLearning/Flow/groundtruth_test_flyingchairs/')


    # eval_flow = np.load(parser.parse_args().flow_file)

    flow_est = load_flow_from_file(parser.parse_args().flowfile)
    print(flow_est.shape[0])
    with open('../flyingchairs_test.txt') as f:
        files = f.readlines()

    aee_total = []
    error_file = open('error.txt', 'w')
    for i in range(len(files)):
        filename = files[i].strip().split()[2]
        flow_gt = read_flow(os.path.join(parser.parse_args().dataset_dir, filename))
        # aee = aee_f(flow_gt, flow_est)
        aee = epe(flow_gt, flow_est[i])
        aee_total.append(aee)
        error_file.write(filename + ' ' + '%.4f\n' % aee)
        print('sample: %d, aee:%.4f' % (i, aee))
        # visualzie_flow(flow_est[i], save_flow=True, save_dir=parser.parse_args().save_dir, file=filename)

    print('mean aee of all test samples: ', np.mean(aee_total))
    error_file.write('mean_aee: %.4f\n' % np.mean(aee_total))
    error_file.close()


if __name__ == '__main__':
    main()
