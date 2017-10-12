# -*- coding: utf-8 -*-
# @Time    : 17-10-2 下午3:04
# @Author  : Fei Xue
# @Email   : feixue@pku.edu.cn
# @File    : model.py
# @Software: PyCharm Community Edition

from collections import namedtuple
import tensorflow as tf
import numpy as np
from skimage.io import imread, imsave
import cv2
import os
import tensorflow.contrib.slim as slim
import tools.visualize as viz
import tools.bilateral_solver as bills

from bilinear_sampler import *

weightedflow_parameters = namedtuple('parameters',
                                     'encoder, '
                                     'height, width, '
                                     'batch_size, '
                                     'batch_norm, '
                                     'record_bytes, '
                                     'd_shape_flow, '
                                     'd_shape_img, '
                                     'num_threads, '
                                     'num_epochs, '
                                     'wrap_mode, '
                                     'use_deconv, '
                                     'alpha_image_loss, '
                                     'flow_gradient_loss_weight, '
                                     'lr_loss_weight, '
                                     'full_summary, '
                                     'scale')


class WeightedFlow(object):
    """weighted optical flow"""

    def __init__(self, params, mode, left, right, flow_gt=None, reuse_variables=None, model_index=0, optim=None, batch_norm=True):
        self.params = params
        self.mode = mode
        self.left = left
        self.right = right
        self.flow_gt = flow_gt
        self.model_collection = ['model_' + str(model_index)]
        self.reuse_variables = reuse_variables
        self.batch_norm = batch_norm

        self.optim = optim
        self.scales = [params.scale, params.scale / 2., params.scale / 4., params.scale / 8.]

        self.build_model()
        self.build_outputs()

        if self.mode == 'test':
            return

        self.build_losses()
        self.build_summaries()

    def gradient_x(self, img):
        gx = img[:, :, :-1, :] - img[:, :, 1:, :]
        return gx

    def gradient_y(self, img):
        gy = img[:, :-1, :, :] - img[:, 1:, :, :]
        return gy

    def upsample_nn(self, x, ratio):
        s = tf.shape(x)
        h = s[1]
        w = s[2]
        return tf.image.resize_nearest_neighbor(x, [h * ratio, w * ratio])

    def scale_pyramid(self, img, num_scales):
        scaled_imgs = [img]
        s = tf.shape(img)
        h = s[1]
        w = s[2]

        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_imgs.append(tf.image.resize_area(img, [nh, nw]))
        return scaled_imgs

    def generate_image_left(self, img, flow):
        return bilinear_sampler(img, -flow)

    def generate_image_right(self, img, flow):
        return bilinear_sampler(img, flow)

    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = slim.avg_pool2d(x, 3, 1, 'VALID')
        mu_y = slim.avg_pool2d(y, 3, 1, 'VALID')

        sigma_x = slim.avg_pool2d(x ** 2, 3, 1, 'VALID') - mu_x ** 2
        sigma_y = slim.avg_pool2d(y ** 2, 3, 1, 'VALID') - mu_y ** 2
        sigma_xy = slim.avg_pool2d(x * y, 3, 1, 'VALID') - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

    def get_flow_smoothness(self, flow, pyramid):
        flow_gradients_x = [self.gradient_x(d) for d in flow]
        flow_gradients_y = [self.gradient_y(d) for d in flow]

        image_gradients_x = [self.gradient_x(img) for img in pyramid]
        image_gradients_y = [self.gradient_y(img) for img in pyramid]

        weights_x = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True)) for g in image_gradients_x]
        weights_y = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True)) for g in image_gradients_y]

        smoothness_x = [flow_gradients_x[i] * weights_x[i] for i in range(4)]
        smoothness_y = [flow_gradients_y[i] * weights_y[i] for i in range(4)]
        return smoothness_x + smoothness_y

    def batch_normalization(self, inputs, is_training=True, decay=0.999, epsilon=1e-3):
        scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
        beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
        pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
        pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

        if is_training:
            batch_mean, batch_var = tf.nn.moments(inputs, [0])
            train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, epsilon)
        else:
            return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)

    def lrelu(self, x, leak=0.1):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

    def conv(self, x, num_out_layers, kernel_size, stride, activation_fn=None, batch_norm=True):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])

        if self.batch_norm and batch_norm:
            output = slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID', activation_fn=activation_fn,
                               normalizer_fn=slim.batch_norm)
        else:
            output = slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID', activation_fn=activation_fn)

        return self.lrelu(output)

    def conv_block(self, x, num_out_layers, kernel_size):
        conv1 = self.conv(x, num_out_layers, kernel_size, 1)
        conv2 = self.conv(conv1, num_out_layers, kernel_size, 2)
        return conv2

    def maxpool(self, x, kernel_size):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.max_pool2d(p_x, kernel_size)

    def resconv(self, x, num_layers, stride):
        do_proj = tf.shape(x)[3] != num_layers or stride == 2
        shortcut = []
        conv1 = self.conv(x, num_layers, 1, 1)
        conv2 = self.conv(conv1, num_layers, 3, stride)
        conv3 = self.conv(conv2, 4 * num_layers, 1, 1, None)
        if do_proj:
            shortcut = self.conv(x, 4 * num_layers, 1, stride, None)
        else:
            shortcut = x
        return self.lrelu(conv3 + shortcut)

    def resblock(self, x, num_layers, num_blocks):
        out = x
        for i in range(num_blocks - 1):
            out = self.resconv(out, num_layers, 1)
        out = self.resconv(out, num_layers, 2)
        return out

    def upconv(self, x, num_out_layers, kernel_size, scale):
        upsample = self.upsample_nn(x, scale)
        conv = self.conv(upsample, num_out_layers, kernel_size, 1)
        return conv

    def deconv(self, x, num_out_layers, kernel_size, scale):
        p_x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
        conv = slim.conv2d_transpose(p_x, num_out_layers, kernel_size, scale, 'SAME')
        conv = self.lrelu(conv)
        return conv[:, 3:-1, 3:-1, :]

    def get_flow(self, x, s, epsilon=1e-7):
        flow = self.conv(x, 4, kernel_size=3, stride=1, activation_fn=None, batch_norm=False)  # negative and positive
        return flow

    def epe(self, input_flow, target_flow):
        square = tf.square(input_flow - target_flow)
        x = square[:, :, :, 0]
        y = square[:, :, :, 1]
        epe = tf.sqrt(tf.add(x, y))
        return epe

    def visualize_flow(self, flow, s):
        square = tf.square(flow)
        x = square[:, :, :, 0]
        y = square[:, :, :, 1]
        sqr = tf.sqrt(tf.add(x, y))
        return sqr
        # return tf.sqrt(tf.multiply(flow[:, :, :, 0], flow[:, :, :, 0]) + tf.multiply(flow[:, :, :, 1], flow[:, :, :, 1]))

    def build_vgg(self):
        # set convenience functions
        conv = self.conv
        if self.params.use_deconv:
            upconv = self.deconv
        else:
            upconv = self.upconv

        with tf.variable_scope('encoder'):
            conv1 = self.conv_block(self.model_input, 32, 7)  # H/2
            conv2 = self.conv_block(conv1, 64, 5)  # H/4
            conv3 = self.conv_block(conv2, 128, 3)  # H/8
            conv4 = self.conv_block(conv3, 256, 3)  # H/16
            conv5 = self.conv_block(conv4, 512, 3)  # H/32
            conv6 = self.conv_block(conv5, 512, 3)  # H/64
            conv7 = self.conv_block(conv6, 512, 3)  # H/128

        with tf.variable_scope('skips'):
            skip1 = conv1
            skip2 = conv2
            skip3 = conv3
            skip4 = conv4
            skip5 = conv5
            skip6 = conv6

        with tf.variable_scope('decoder'):
            upconv7 = upconv(conv7, 512, 3, 2)  # H/64
            concat7 = tf.concat([upconv7, skip6], 3)
            iconv7 = conv(concat7, 512, 3, 1)

            upconv6 = upconv(iconv7, 512, 3, 2)  # H/32
            concat6 = tf.concat([upconv6, skip5], 3)
            iconv6 = conv(concat6, 512, 3, 1)

            upconv5 = upconv(iconv6, 256, 3, 2)  # H/16
            concat5 = tf.concat([upconv5, skip4], 3)
            iconv5 = conv(concat5, 256, 3, 1)

            upconv4 = upconv(iconv5, 128, 3, 2)  # H/8
            concat4 = tf.concat([upconv4, skip3], 3)
            iconv4 = conv(concat4, 128, 3, 1)
            self.flow4 = self.get_flow(iconv4, 3)
            uflow4 = self.upsample_nn(self.flow4, 2)

            upconv3 = upconv(iconv4, 64, 3, 2)  # H/4
            concat3 = tf.concat([upconv3, skip2, uflow4], 3)
            iconv3 = conv(concat3, 64, 3, 1)
            self.flow3 = self.get_flow(iconv3, 2)
            uflow3 = self.upsample_nn(self.flow3, 2)

            upconv2 = upconv(iconv3, 32, 3, 2)  # H/2
            concat2 = tf.concat([upconv2, skip1, uflow3], 3)
            iconv2 = conv(concat2, 32, 3, 1)
            self.flow2 = self.get_flow(iconv2, 1)
            uflow2 = self.upsample_nn(self.flow2, 2)

            upconv1 = upconv(iconv2, 16, 3, 2)  # H
            concat1 = tf.concat([upconv1, uflow2], 3)
            iconv1 = conv(concat1, 16, 3, 1)
            self.flow1 = self.get_flow(iconv1, 0)

    def build_resnet50(self):
        # set convenience functions
        conv = self.conv
        if self.params.use_deconv:
            upconv = self.deconv
        else:
            upconv = self.upconv

        with tf.variable_scope('encoder'):
            conv1 = conv(self.model_input, 64, 7, 2)  # H/2  -   64D
            pool1 = self.maxpool(conv1, 3)  # H/4  -   64D
            conv2 = self.resblock(pool1, 64, 3)  # H/8  -  256D
            conv3 = self.resblock(conv2, 128, 4)  # H/16 -  512D
            conv4 = self.resblock(conv3, 256, 6)  # H/32 - 1024D
            conv5 = self.resblock(conv4, 512, 3)  # H/64 - 2048D

        with tf.variable_scope('skips'):
            skip1 = conv1
            skip2 = pool1
            skip3 = conv2
            skip4 = conv3
            skip5 = conv4

        # DECODING
        with tf.variable_scope('decoder'):
            upconv6 = upconv(conv5, 512, 3, 2)  # H/32
            concat6 = tf.concat([upconv6, skip5], 3)
            iconv6 = conv(concat6, 512, 3, 1)

            upconv5 = upconv(iconv6, 256, 3, 2)  # H/16
            concat5 = tf.concat([upconv5, skip4], 3)
            iconv5 = conv(concat5, 256, 3, 1)

            upconv4 = upconv(iconv5, 128, 3, 2)  # H/8
            concat4 = tf.concat([upconv4, skip3], 3)
            iconv4 = conv(concat4, 128, 3, 1)
            self.flow4 = self.get_flow(iconv4, 3)
            uflow4 = self.upsample_nn(self.flow4, 2)

            upconv3 = upconv(iconv4, 64, 3, 2)  # H/4
            concat3 = tf.concat([upconv3, skip2, uflow4], 3)
            iconv3 = conv(concat3, 64, 3, 1)
            self.flow3 = self.get_flow(iconv3, 2)
            uflow3 = self.upsample_nn(self.flow3, 2)

            upconv2 = upconv(iconv3, 32, 3, 2)  # H/2
            concat2 = tf.concat([upconv2, skip1, uflow3], 3)
            iconv2 = conv(concat2, 32, 3, 1)
            self.flow2 = self.get_flow(iconv2, 1)
            uflow2 = self.upsample_nn(self.flow2, 2)

            upconv1 = upconv(iconv2, 16, 3, 2)  # H
            concat1 = tf.concat([upconv1, uflow2], 3)
            iconv1 = conv(concat1, 16, 3, 1)
            self.flow1 = self.get_flow(iconv1, 0)

    def build_model(self):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=tf.nn.elu):
            with tf.variable_scope('model', reuse=self.reuse_variables):
                self.left_pyramid = self.scale_pyramid(self.left, 4)
                self.right_pyramid = self.scale_pyramid(self.right, 4)

                self.model_input = tf.concat([self.left, self.right], 3)

                self.flow_gt_pyramid = self.scale_pyramid(self.flow_gt, 4)

                # build model
                if self.params.encoder == 'vgg':
                    self.build_vgg()
                elif self.params.encoder == 'resnet50':
                    self.build_resnet50()
                else:
                    return None

    def build_outputs(self):
        # Store Flow
        with tf.variable_scope('flows'):
            self.flow_est = [self.flow1, self.flow2, self.flow3, self.flow4]
            self.flow_left_est = [d[:, :, :, 0:2] for d in self.flow_est]
            self.flow_right_est = [d[:, :, :, 2:4] for d in self.flow_est]

            self.flow_images = [self.visualize_flow(self.flow_left_est[i], i) for i in range(4)]

            self.flow_final = (self.flow_left_est[0] + self.flow_right_est[0]) / 2
            self.flow_diff = self.epe(self.flow_final, self.flow_gt)
            self.flow_error = tf.reduce_mean(self.flow_diff)
            self.flow_gt_img = self.visualize_flow(self.flow_gt, 0)

        if self.mode == 'test':
            # self.flow_left_est[0][: ,:, :, 0] *= self.params.width
            # self.flow_left_est[0][: ,:, :, 1] *= self.params.height

            # self.flow_left_est[0] = tf.image.resize_bicubic(self.flow_left_est[0], [384, 512])
            # self.flow_right_est[0] = tf.image.resize_bicubic(self.flow_left_est[0], [384, 512])
            return

        # Generate images
        with tf.variable_scope('images'):
            self.left_est = [self.generate_image_left(self.right_pyramid[i], self.flow_left_est[i]) for i in range(4)]
            self.right_est = [self.generate_image_right(self.left_pyramid[i], self.flow_right_est[i]) for i in range(4)]

        # LR consistency
        with tf.variable_scope('left-right'):
            self.right_to_left_flow = [self.generate_image_left(self.flow_right_est[i], self.flow_left_est[i]) for i in
                                       range(4)]
            self.left_to_right_flow = [self.generate_image_right(self.flow_left_est[i], self.flow_right_est[i]) for i in
                                       range(4)]

        # Flow smoothness
        with tf.variable_scope('smoothness'):
            self.flow_left_smoothness = self.get_flow_smoothness(self.flow_left_est, self.left_pyramid)
            self.flow_right_smoothness = self.get_flow_smoothness(self.flow_right_est, self.right_pyramid)

    def build_losses(self):
        with tf.variable_scope('losses', reuse=self.reuse_variables):
            # Image reconstruction
            # L1
            self.l1_left = [tf.abs(self.left_est[i] - self.left_pyramid[i]) for i in range(4)]
            self.l1_reconstruction_loss_left = [tf.reduce_mean(l) for l in self.l1_left]
            self.l1_right = [tf.abs(self.right_est[i] - self.right_pyramid[i]) for i in range(4)]
            self.l1_reconstruction_loss_right = [tf.reduce_mean(l) for l in self.l1_right]

            # SSIM
            self.ssim_left = [self.SSIM(self.left_est[i], self.left_pyramid[i]) for i in range(4)]
            self.ssim_loss_left = [tf.reduce_mean(s) for s in self.ssim_left]
            self.ssim_right = [self.SSIM(self.right_est[i], self.right_pyramid[i]) for i in range(4)]
            self.ssim_loss_right = [tf.reduce_mean(s) for s in self.ssim_right]

            # Weight Sum
            self.image_loss_right = [
                self.params.alpha_image_loss * self.ssim_loss_right[i] + (1 - self.params.alpha_image_loss) *
                self.l1_reconstruction_loss_right[i] for i in range(4)]
            self.image_loss_left = [
                self.params.alpha_image_loss * self.ssim_loss_left[i] + (1 - self.params.alpha_image_loss) *
                self.l1_reconstruction_loss_left[i] for i in range(4)]
            self.image_loss = tf.add_n(self.image_loss_left + self.image_loss_right)

            # Flow smoothness
            self.flow_left_loss = [tf.reduce_mean(tf.abs(self.flow_left_smoothness[i])) / 2 ** i for i in range(4)]
            self.flow_right_loss = [tf.reduce_mean(tf.abs(self.flow_right_smoothness[i])) / 2 ** i for i in range(4)]
            self.flow_gradient_loss = tf.add_n(self.flow_left_loss + self.flow_right_loss)

            # LR consistency
            self.lr_left_loss = [tf.reduce_mean(tf.abs(self.right_to_left_flow[i] - self.flow_left_est[i])) for i in
                                 range(4)]
            self.lr_right_loss = [tf.reduce_mean(tf.abs(self.left_to_right_flow[i] - self.flow_right_est[i])) for i in
                                  range(4)]
            self.lr_loss = tf.add_n(self.lr_left_loss + self.lr_right_loss)

            # Total loss
            self.total_loss = self.image_loss + self.params.flow_gradient_loss_weight * self.flow_gradient_loss + self.params.lr_loss_weight * self.lr_loss

    def build_summaries(self):
        # SUMMARIES
        max_output = 3
        with tf.device('/cpu:0'):
            for i in range(4):
                tf.summary.scalar('ssim_loss_' + str(i), self.ssim_loss_left[i] + self.ssim_loss_right[i],
                                  collections=self.model_collection)
                tf.summary.scalar('l1_loss_' + str(i),
                                  self.l1_reconstruction_loss_left[i] + self.l1_reconstruction_loss_right[i],
                                  collections=self.model_collection)
                tf.summary.scalar('image_loss_' + str(i), self.image_loss_left[i] + self.image_loss_right[i],
                                  collections=self.model_collection)
                tf.summary.scalar('flow_gradient_loss_' + str(i), self.flow_left_loss[i] + self.flow_right_loss[i],
                                  collections=self.model_collection)
                tf.summary.scalar('lr_loss_' + str(i), self.lr_left_loss[i] + self.lr_right_loss[i],
                                  collections=self.model_collection)
                tf.summary.image('flow_left_est_' + str(i),
                                 tf.expand_dims(tf.abs(self.flow_left_est[i][:, :, :, 0]), -1), max_outputs=max_output,
                                 collections=self.model_collection)
                tf.summary.image('flow_left_est_' + str(i),
                                 tf.expand_dims(tf.abs(self.flow_left_est[i][:, :, :, 1]), -1), max_outputs=max_output,
                                 collections=self.model_collection)
                tf.summary.image('flow_right_est_' + str(i),
                                 tf.expand_dims(tf.abs(self.flow_right_est[i][:, :, :, 0]), -1), max_outputs=max_output,
                                 collections=self.model_collection)
                tf.summary.image('flow_right_est_' + str(i),
                                 tf.expand_dims(tf.abs(self.flow_right_est[i][:, :, :, 1]), -1), max_outputs=max_output,
                                 collections=self.model_collection)
                tf.summary.image('flow_result_' + str(i), tf.expand_dims(self.flow_images[i], -1),
                                 max_outputs=max_output, collections=self.model_collection)
                # tf.summary.image('_result_' + str(i), tf.expand_dims(self.flow_images[i], -1), max_outputs=max_output, collections=self.model_collection)
                # tf.summary.image('flow_result_' + str(i), tf.expand_dims(self.flow_images[i], -1), max_outputs=max_output, collections=self.model_collection)

                if self.params.full_summary:
                    tf.summary.image('left_est_' + str(i), self.left_est[i], max_outputs=max_output,
                                     collections=self.model_collection)
                    tf.summary.image('right_est_' + str(i), self.right_est[i], max_outputs=max_output,
                                     collections=self.model_collection)
                    tf.summary.image('ssim_left_' + str(i), self.ssim_left[i], max_outputs=max_output,
                                     collections=self.model_collection)
                    tf.summary.image('ssim_right_' + str(i), self.ssim_right[i], max_outputs=max_output,
                                     collections=self.model_collection)
                    tf.summary.image('l1_left_' + str(i), self.l1_left[i], max_outputs=max_output,
                                     collections=self.model_collection)
                    tf.summary.image('l1_right_' + str(i), self.l1_right[i], max_outputs=max_output,
                                     collections=self.model_collection)
                    tf.summary.image('left_' + str(i), self.left_pyramid[i], max_outputs=max_output,
                                     collections=self.model_collection)
                    tf.summary.image('right_' + str(i), self.right_pyramid[i], max_outputs=max_output,
                                     collections=self.model_collection)
            tf.summary.image('flow_error_with_gt', tf.expand_dims(self.flow_diff, -1), max_outputs=max_output, collections=self.model_collection)
            tf.summary.image('flow_groundtruth', tf.expand_dims(self.flow_gt_img, -1), max_outputs=max_output, collections=self.model_collection)
                    # if self.optim is not None:
                    #     train_vars = [var for var in tf.trainable_variables()]
                    #     self.grads_and_vars = self.optim.compute_gradients(self.total_loss,
                    #                                                   var_list=train_vars)
                    #     for var in tf.trainable_variables():
                    #           tf.summary.histogram(var.op.name + "/values", var)
                    #     for grad, var in self.grads_and_vars:
                    #         tf.summary.histogram(var.op.name + "/gradients", grad)

    def build_test(self):
        return self.left_est[0]
