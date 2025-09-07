import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import time

def set_figsize(figsize=(5.5, 3.5)):  
    plt.rcParams['figure.figsize'] = figsize

def set_axes(xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴"""
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.xlim(xlim)
    plt.ylim(ylim)
    if legend:
        plt.legend(legend)
    plt.grid()


def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(5.5, 3.5), axes=None):
    """绘制数据点"""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else plt.gca()

    # 如果X有一个轴，输出True
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    plt.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            plt.plot(y, fmt)
    set_axes(xlabel, ylabel, xlim, ylim, xscale, yscale, legend)

def construct_data(w, b, num_samples): 
    x = tf.zeros((num_samples, w.shape[0]))
    x += tf.random.normal(x.shape)
    y = tf.matmul(x, tf.reshape(w, (-1, 1))) + b
    y += tf.random.normal(y.shape, stddev=0.01)
    y = tf.reshape(y, (-1, 1))
    return x,y

def data_iter(batch_size, features, labels):
    num = len(features)
    indices = list(range(0, num))
    random.shuffle(indices)
    for j in range(0, num, batch_size):
        k = indices[j:min(j + batch_size, num)]
        yield tf.gather(features, k), tf.gather(labels, k)

#定义优化函数
def sgd(params, grads, lr, batch_size):
    for param,grad in zip(params, grads):
        param.assign_sub(lr*grad/batch_size)

def load_array(data_arrays, batch_size, shuffle_size, is_train=True):
    dataset = tf.data.Dataset.from_tensor_slices(data_arrays)
    if is_train:
        dataset = dataset.shuffle(shuffle_size)
    dataset = dataset.batch(batch_size)
    return dataset

def gpu_mem_init():
    # 列出所有可用的物理 GPU 设备
    gpus = tf.config.list_physical_devices('GPU')

    # 如果有可用的 GPU，启用显存增长
    if gpus:
        # 为每个 GPU 设备设置显存增长选项
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # 确认设置结果
    for gpu in gpus:
        growth_setting = tf.config.experimental.get_memory_growth(gpu)
        print(f"Memory growth enabled for {gpu}: {growth_setting}")

class Timer:  #@save
    """记录多次运行时间"""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()

def load_data_fashion_mnist(batch_size, resize=None):
    mi_train,mi_test = tf.keras.datasets.fashion_mnist.load_data()
    process = lambda x,y : (tf.expand_dims(x, axis=2)/255, tf.cast(y, dtype='int32'))
    resize_fn = lambda x,y : (tf.image.resize_with_pad(x, resize, resize) if resize else x,y)
    return (tf.data.Dataset.from_tensor_slices(process(*mi_train)).shuffle(len(mi_train[0])).batch(batch_size).map(resize_fn),
            tf.data.Dataset.from_tensor_slices(process(*mi_test)).batch(batch_size).map(resize_fn))