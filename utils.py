import pandas as pd
import matplotlib.pyplot as plt


def plot_gallery(title, images, n_col, n_row, sm_shape):
    '''
    绘图函数
    :param title: 图像名称
    :param images: 图像
    :param n_col: 输出多少个子图，对应有n_col*n_row个子图
    :param n_row:
    :return:
    '''
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        plt.imshow(comp.reshape(sm_shape).T, cmap=plt.cm.gray,
                   interpolation='nearest',
                   vmin=-vmax, vmax=vmax)
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.)