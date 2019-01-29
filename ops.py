import numpy as np

import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, Conv2D, SeparableConv2D, MaxPool2D, AveragePooling2D
from tensorflow.python.keras.layers import BatchNormalization, GlobalAveragePooling2D


class Identity(Model):

    def __init__(self, filters, strides):
        super(Identity, self).__init__()

        if strides == (2, 2):
            self.op = Conv2D(filters, (1, 1), strides, padding='same',
                             kernel_initializer='he_uniform')
        else:
            self.op = lambda x: x

    def call(self, inputs, training=None, mask=None):
        return self.op(inputs)


class SeperableConvolution(Model):

    def __init__(self, filters, kernel, strides):
        super(SeperableConvolution, self).__init__()

        self.conv = SeparableConv2D(filters, kernel, strides=strides, padding='same',
                                    kernel_initializer='he_uniform')
        self.bn = BatchNormalization()

    def call(self, inputs, training=None, mask=None):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        return tf.nn.relu(x)


class Convolution(Model):

    def __init__(self, filters, kernel, strides):
        super(Convolution, self).__init__()

        self.conv = Conv2D(filters, kernel, strides=strides, padding='same',
                           kernel_initializer='he_uniform')
        self.bn = BatchNormalization()

    def call(self, inputs, training=None, mask=None):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        return tf.nn.relu(x)


class StackedConvolution(Model):

    def __init__(self, filter_list, kernel_list, stride_list):
        super(StackedConvolution, self).__init__()

        assert len(filter_list) == len(kernel_list) and len(kernel_list) == len(stride_list), "List lengths must match"

        self.convs = []
        for i, (f, k, s) in enumerate(zip(filter_list, kernel_list, stride_list)):
            conv = Convolution(f, k, s)

            self.convs.append(conv)

    def call(self, inputs, training=None, mask=None):
        x = inputs

        for conv in self.convs:
            x = conv(x, training=training)

        return x


class Pooling(Model):

    def __init__(self, type, size, strides):
        super(Pooling, self).__init__()

        if type == 'max':
            self.pool = MaxPool2D(size, strides, padding='same')
        else:
            self.pool = AveragePooling2D(size, strides, padding='same')

    def call(self, inputs, training=None, mask=None):
        return self.pool(inputs)
