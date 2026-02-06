import tensorflow as tf
import sys
import math
import numpy as np
sys.path.append('../../pytools')
import d2l

d2l.gpu_mem_init()

def vgg_block(num_convs, num_channels):
    blk = tf.keras.models.Sequential()
    for _ in range(num_convs):
        blk.add(tf.keras.layers.Conv2D(num_channels, kernel_size=3, padding='same', activation='relu'))
    blk.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    return blk

conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
def vgg(conv_arch):
    net = tf.keras.models.Sequential()
    for num_conv,num_channel in conv_arch:
        net.add(vgg_block(num_conv, num_channel))
    net.add(tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10)
    ]))
    return net

ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]

vgg_net_0 = lambda: vgg(small_conv_arch)

lr, num_epochs, batch_size = 0.05, 5, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(vgg_net_0, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())