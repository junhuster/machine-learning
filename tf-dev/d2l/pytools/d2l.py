import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import time
from IPython import display

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

def linreg(X, w, b):  #@save
    """线性回归模型"""
    return tf.matmul(X, w) + b

def squared_loss(y_hat, y):  #@save
    """均方损失"""
    return (y_hat - tf.reshape(y, y_hat.shape)) ** 2 / 2

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
    process = lambda x,y : (tf.expand_dims(x, axis=3)/255, tf.cast(y, dtype='int32'))
    resize_fn = lambda x,y : (tf.image.resize_with_pad(x, resize, resize) if resize else x,y)
    return (tf.data.Dataset.from_tensor_slices(process(*mi_train)).shuffle(len(mi_train[0])).batch(batch_size).map(resize_fn),
            tf.data.Dataset.from_tensor_slices(process(*mi_test)).batch(batch_size).map(resize_fn))

def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """训练模型一个迭代周期（定义见第3章）"""
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        with tf.GradientTape() as tape:
            y_hat = net(X)
            # Keras内置的损失接受的是（标签，预测），这不同于用户在本书中的实现。
            # 本书的实现接受（预测，标签），例如我们上面实现的“交叉熵”
            if isinstance(loss, tf.keras.losses.Loss):
                l = loss(y, y_hat)
            else:
                l = loss(y_hat, y)
        if isinstance(updater, tf.keras.optimizers.Optimizer):
            params = net.trainable_variables
            grads = tape.gradient(l, params)
            updater.apply_gradients(zip(grads, params))
        else:
            updater(X.shape[0], tape.gradient(l, updater.params))
        # Keras的loss默认返回一个批量的平均损失
        l_sum = l * float(tf.size(y)) if isinstance(
            loss, tf.keras.losses.Loss) else tf.reduce_sum(l)
        metric.add(l_sum, accuracy(y_hat, y), tf.size(y))
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        train_loss, train_acc = train_metrics
        animator.add(epoch + 1, train_metrics + (test_acc,))
        print(f'epoch:{epoch}, train_loss:{train_loss:f}, train_auc:{train_acc:f}')

class Updater():  #@save
    """用小批量随机梯度下降法更新参数"""
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def __call__(self, batch_size, grads):
        sgd(self.params, grads, self.lr, batch_size)

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = tf.argmax(y_hat, axis=1)
    cmp = tf.cast(y_hat, dtype=y.dtype) == y
    return float(tf.reduce_sum(tf.cast(cmp, y.dtype)))

class Accumulator:  #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def evaluate_accuracy(net, data_iter):
    metric = Accumulator(2)
    for x,y in data_iter:
        metric.add(accuracy(net(x), y), len(y))
    return metric[0] / metric[1]

def show_images(imgs, num_rows, num_cols, titles=None, scale=2):  #@save
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(img.numpy())
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

class Animator:  #@save
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: set_axes(
            xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)

def get_fashion_mnist_labels(labels):  #@save
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def predict_ch3(net, mi_test, n = 80):
    for x,y in mi_test:
        break
    trues = get_fashion_mnist_labels(y)
    preds = get_fashion_mnist_labels(tf.argmax(net(x), axis=1))
    titles = [true + '\n' + pred for true,pred in zip(trues, preds)]
    show_images(tf.reshape(x[0:n], (n, 28, 28)), 1, 8, titles[0:n])

def evaluate_loss(net, data_iter, loss):
    metric = Accumulator(2)
    for x,y in data_iter:
        l = loss(net(x), y)
        metric.add(tf.reduce_sum(l), tf.size(y))
    return metric[0] / metric[1]