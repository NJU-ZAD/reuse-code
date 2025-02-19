#!/usr/bin/python3
# -*- coding:utf-8 -*-
import copy
import math
import os
import random

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as scinpo
import seaborn as sns
from matplotlib import cm
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Ellipse
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
from numpy.lib.scimath import logn
from sklearn.metrics import r2_score


def plot_scatter(x_data, y_data, save_path, x_label, y_label,
                 var_orient='horizon'):
    """
    绘制散点图
    """
    if var_orient != 'vertical' and var_orient != 'horizon':
        print('plot_figure var_orient error!')
        exit(1)
    if var_orient == 'vertical':
        y_data = list(map(list, zip(*y_data)))
    x_number = len(y_data[0])
    y_number = len(y_data)
    if len(x_data) != x_number:
        print('x_data和y_data不匹配！')
        exit(1)
    # 设置y_data的label、color、linestyle、marker
    if y_number == 1:
        label = ['label1']
    elif y_number == 2:
        label = ['label1', 'label2']
    elif y_number == 3:
        label = ['label1', 'label2', 'label3']
    else:
        label = []
    # HTML颜色名 'red', 'green', 'blue', 'yellow', 'black', 'white',
    # 'cyan', 'darksalmon', 'gold', 'crimson'
    if y_number == 1:
        color = ['red']
    elif y_number == 2:
        color = ['red', 'green']
    elif y_number == 3:
        color = ['red', 'green', 'black']
    else:
        color = []
    # 线条风格 '-', '--', '-.', ':', 'None'
    if y_number == 1:
        linestyle = ['-']
    elif y_number == 2:
        linestyle = ['-', '--']
    elif y_number == 3:
        linestyle = ['-', '--', '-.']
    else:
        linestyle = []
    # 线条标记 'o', 'd', 'D', 'h', 'H', '_', '8', 'p', ',', '.', 's',
    #  '+', '*', 'x', '^', 'v', '<', '>', '|'
    if y_number == 1:
        marker = ['o']
    elif y_number == 2:
        marker = ['v', '^']
    elif y_number == 3:
        marker = ['v', '^', '.']
    else:
        marker = []
    if y_number != len(label) or y_number != len(color) or \
            y_number != len(linestyle) or y_number != len(marker):
        print('y_data的label、color、linestyle、marker没有被正确设置！')
        exit(1)
    # 设置标题
    # plt.title('title')
    # 设置字体和大小
    plt.rc('text', usetex=False)
    plt.rc('font', family='Times New Roman', size=15)
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.bf'] = 'Times New Roman'
    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}
    # 获取当前Axes对象
    ax = plt.gca()
    # 设置x轴的主刻度
    ax.xaxis.set_major_locator(MultipleLocator(1))
    # 设置x轴的副刻度
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    # 设置y轴的主刻度
    ax.yaxis.set_major_locator(MultipleLocator(5))
    # 设置y轴的副刻度
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    # 以科学计数法表示y轴数据
    ax.ticklabel_format(
        style='sci', scilimits=(0, 1), axis='y', useMathText=True)
    # 设置x轴的刻度标签
    plt.xticks(x_data, x_data, rotation=0)
    # 设置y轴的刻度标签
    plt.yticks(rotation=15)
    # 设置x轴的标签
    plt.xlabel(x_label, fontweight='bold', size=15)
    # 设置x轴的范围
    # plt.xlim([0, 5])
    # 设置y轴的标签
    plt.ylabel(y_label, fontweight='bold', size=15)
    # 设置y轴的范围
    plt.ylim([0, 25])
    types = []
    for _ in range(y_number):
        # 'none'表示无填充颜色
        res = plt.scatter(
            x_data, y_data[_], color=color[_], edgecolors=color[_],
            marker=marker[_], s=200, linewidths=3, alpha=0.4)
        types.append(res)

    # 设置网格线
    plt.grid(axis='both', color='gray', linestyle='-')
    plt.legend(tuple(types), tuple(label), loc=2, ncol=2, prop=font1)
    fig = plt.gcf()
    # 设置图片的长和宽（英寸）
    fig.set_size_inches(9, 3.5)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print('已将图像保存为'+save_path)
    plt.show()


def plot_ellipse(x_min, x_max, y_min, y_max, save_path, x_label, y_label):
    """
    绘制椭圆分布图
    """
    if not (len(x_min) == len(x_max) == len(y_min) == len(y_max)):
        print('x_min、x_max、y_min和y_max不匹配！')
        exit(1)
    number = len(x_min)
    # 设置label、color、marker、alpha、xtext、ytext
    if number == 1:
        label = ['label1']
    elif number == 2:
        label = ['label1', 'label2']
    elif number == 3:
        label = ['label1', 'label2', 'label3']
    else:
        label = ['label1', 'label2', 'label3', 'label4', 'label5']
    # HTML颜色名 'red', 'green', 'blue', 'yellow', 'black', 'white',
    # 'cyan', 'darksalmon', 'gold', 'crimson'
    if number == 1:
        color = ['red']
    elif number == 2:
        color = ['red', 'green']
    elif number == 3:
        color = ['red', 'green', 'black']
    else:
        color = ['red', 'green', 'black', 'blue', 'yellow']
    # 散点标记 'o', 'd', 'D', 'h', 'H', '_', '8', 'p', ',', '.', 's',
    #  '+', '*', 'x', '^', 'v', '<', '>', '|'
    if number == 1:
        marker = ['o']
    elif number == 2:
        marker = ['v', '^']
    elif number == 3:
        marker = ['v', '^', '.']
    else:
        marker = ['.', '.', '.', '.', '.']
    alpha = [0.4, 0.4, 0.4, 0.4, 0.4]
    xtext = [0, 0, 0, 0, 0]
    ytext = [1.5, 1.5, -1.5, -1.5, 1.5]
    # 方差越大数据越发散越接近椭圆
    # 方差越小数据越稳定越接近直线
    variance = [1, 1, 1, 1, 1]
    assert all(0 < var <= 1 for var in variance)
    if number != len(label) or number != len(color) or \
            number != len(marker) or number != len(alpha) or \
        number != len(xtext) or number != len(ytext):
        print('label、color、marker、alpha、xtext、ytext没有被正确设置！')
        exit(1)
    # 设置标题
    # plt.title('title')
    # 设置字体和大小
    plt.rc('text', usetex=False)
    plt.rc('font', family='Times New Roman', size=15)
    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}
    # 获取当前Axes对象
    ax = plt.gca()
    # 设置x轴的主刻度
    ax.xaxis.set_major_locator(MultipleLocator(1))
    # 设置y轴的主刻度
    ax.yaxis.set_major_locator(MultipleLocator(5))
    # 设置x轴的刻度标签
    plt.xticks(rotation=0)
    # 设置y轴的刻度标签
    plt.yticks(rotation=0)
    # 设置x轴的标签
    plt.xlabel(x_label, fontweight='bold', size=15)
    # 设置x轴的范围
    plt.xlim([1, 10])
    # 设置y轴的标签
    plt.ylabel(y_label, fontweight='bold', size=15)
    # 设置y轴的范围
    plt.ylim([70, 100])
    for _ in range(number):
        x = (x_min[_]+x_max[_])/2
        y = (y_min[_]+y_max[_])/2
        w = x_max[_] - x_min[_]
        h = y_max[_] - y_min[_]
        if w != 0:
            r = math.atan(h/w)*180/math.pi
            width = math.sqrt(w**2+h**2)
            height = min(w, h) * variance[_]
            ax.add_artist(Ellipse((x, y), width=width, height=height,
                                  angle=r, edgecolor='None', fc=color[_], alpha=alpha[_]))
        plt.scatter(x, y, color='k', edgecolors='k', marker=marker[_])
        plt.text(x+xtext[_], y+ytext[_], s=label[_],
                 ha='center', va='center', fontsize=15, color='k')

    arrow_start = (10, 70)
    arrow_direction = (-2, 5)
    arrow_props = dict(facecolor='white', edgecolor='black',
                       linewidth=1, width=10, headwidth=18, headlength=25)
    plt.annotate(None, xy=(arrow_start[0]+arrow_direction[0], arrow_start[1] +
                 arrow_direction[1]), xytext=arrow_start, arrowprops=arrow_props)
    text_position = (arrow_start[0] + arrow_direction[0] / 2,
                     arrow_start[1] + arrow_direction[1] / 2)
    x_inches = 9
    y_inches = 3.5
    x_lim = ax.get_xlim()[1]-ax.get_xlim()[0]
    y_lim = ax.get_ylim()[1]-ax.get_ylim()[0]
    scale = (x_lim/y_lim)*(y_inches/x_inches)
    arrow_angle = math.atan(
        scale*arrow_direction[1]/arrow_direction[0])*180/math.pi
    ax.text(*text_position, 'Better',
            ha='center', va='center', color='black', rotation=arrow_angle)

    # 设置网格线
    plt.grid(axis='both', color='gray', linestyle='-')
    fig = plt.gcf()
    # 设置图片的长和宽（英寸）
    fig.set_size_inches(x_inches, y_inches)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print('已将图像保存为'+save_path)
    plt.show()


def get_smooth_xy(x, y, density=50):
    """
    对散点进行平滑处理
    """
    smooth_x = np.linspace(min(x), max(x), density)
    smooth_y = scinpo.make_interp_spline(x, y)(smooth_x)
    return smooth_x, smooth_y


def plot_line(x_data, y_data, save_path, x_label, y_label,
              var_orient='horizon', broken_y_axis=False, latex=True):
    """
    绘制折线图
    """
    if var_orient != 'vertical' and var_orient != 'horizon':
        print('plot_figure var_orient error!')
        exit(1)
    if var_orient == 'vertical':
        y_data = list(map(list, zip(*y_data)))
    x_number = len(y_data[0])
    y_number = len(y_data)
    if len(x_data) != x_number:
        print('x_data和y_data不匹配！')
        exit(1)
    # 设置x_data对应的标签
    x_data_label = [r'$\boldsymbol{\alpha}$', r'$\boldsymbol{2^1}$',
                    r'$\boldsymbol{\beta}$', r'$\boldsymbol{2^2}$',
                    r'\textbf{data24}', r'$\boldsymbol{\epsilon}$',
                    r'$\boldsymbol{2^4}$']
    if len(x_data_label) % x_number != 0:
        print('x_data对应的标签没有被正确设置！')
        exit(1)
    # 设置y_data的label、color、linestyle、marker
    if y_number == 1:
        label = ['']
    elif y_number == 2:
        label = ['label1', 'label2']
    elif y_number == 3:
        label = [r'\textbf{label1}', r'\textbf{label2}', r'\textbf{label3}']
    else:
        label = []
    # HTML颜色名 'red', 'green', 'blue', 'yellow', 'black', 'white',
    # 'cyan', 'darksalmon', 'gold', 'crimson'
    if y_number == 1:
        color = ['red']
    elif y_number == 2:
        color = ['red', 'green']
    elif y_number == 3:
        color = ['red', 'green', 'black']
    else:
        color = []
    # 线条风格 '-', '--', '-.', ':', 'None'
    if y_number == 1:
        linestyle = ['-']
    elif y_number == 2:
        linestyle = ['-', '--']
    elif y_number == 3:
        linestyle = ['-', '--', '-.']
    else:
        linestyle = []
    # 线条标记 'o', 'd', 'D', 'h', 'H', '_', '8', 'p', ',', '.', 's',
    #  '+', '*', 'x', '^', 'v', '<', '>', '|'
    if y_number == 1:
        marker = ['o']
    elif y_number == 2:
        marker = ['v', '^']
    elif y_number == 3:
        marker = ['v', '^', 'o']
    else:
        marker = []
    if y_number != len(label) or y_number != len(color) or \
            y_number != len(linestyle) or y_number != len(marker):
        print('y_data的label、color、linestyle、marker没有被正确设置！')
        exit(1)
    # 设置标题
    # plt.title('title')
    # 设置字体和大小
    if latex is True:
        plt.rc('text', usetex=True)
        mpl.rc('text.latex', preamble=r'\usepackage{amsmath}')
    plt.rc('font', family='Times New Roman', size=15)
    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}
    if broken_y_axis is True:
        fig, (ax_top, ax_bottom) = plt.subplots(2, 1)
    # 获取当前Axes对象
    ax = plt.gca()
    # 设置x轴的主刻度
    ax.xaxis.set_major_locator(MultipleLocator(1))
    # 设置x轴的副刻度
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    # 设置x轴的刻度标签
    # plt.xticks(x_data, x_data, rotation=0)
    plt.xticks(x_data, x_data_label, rotation=0)
    if broken_y_axis is False:
        # 设置y轴的主刻度
        ax.yaxis.set_major_locator(MultipleLocator(5))
        # 设置y轴的副刻度
        ax.yaxis.set_minor_locator(MultipleLocator(1))
        # 设置x轴的刻度标签
        plt.xticks(rotation=15)
        # 设置y轴的刻度标签
        plt.yticks(rotation=15)
        # 设置x轴的标签
        plt.xlabel(x_label, fontweight='bold', size=15)
        # 设置x轴的范围
        # plt.xlim([0, 5])
        # 设置y轴的标签
        plt.ylabel(y_label, fontweight='bold', size=15)
        # 设置y轴的范围
        plt.ylim([0, 25])
    else:
        # 设置y轴断裂处上方的范围
        ylim_top = [15, 25]
        ax_top.set_ylim(ylim_top)
        # 设置y轴断裂处下方的范围
        ylim_bottom = [0, 10]
        ax_bottom.set_ylim(ylim_bottom)
        # 设置y轴的主副刻度
        ax_top.yaxis.set_major_locator(MultipleLocator(5))
        ax_top.yaxis.set_minor_locator(MultipleLocator(1))
        ax_bottom.yaxis.set_major_locator(MultipleLocator(5))
        ax_bottom.yaxis.set_minor_locator(MultipleLocator(1))
        # 设置y轴断裂的距离
        plt.subplots_adjust(hspace=0.15)
        # 设置x轴的刻度标签
        plt.xticks(rotation=15)
        # 设置y轴的刻度标签
        ax_top.tick_params(labelrotation=15)
        plt.yticks(rotation=15)
        # 实现y轴的断裂
        ax_top.spines['bottom'].set_visible(False)
        ax_bottom.spines['top'].set_visible(False)
        ax_top.set_xticks([])
        # 设置x轴和y轴的标签
        ax_bottom.set_xlabel(x_label, fontweight='bold', size=15)
        ax_bottom.set_ylabel(y_label, fontweight='bold', size=15)
        # 设置y轴的标签的位置
        ax_bottom.yaxis.set_label_coords(0.05, 0.5, transform=fig.transFigure)

    if y_number == 3:
        # 为第三条（编号为2）曲线的y值设定上下波动范围
        max_y_var = [y_data[2][_] + (0.5+3*random.random())
                     for _ in range(x_number)]
        min_y_var = [y_data[2][_] - (0.5+2*random.random())
                     for _ in range(x_number)]
        sx, smaxy = get_smooth_xy(x_data, max_y_var)
        sx, sminy = get_smooth_xy(x_data, min_y_var)

    for _ in range(y_number):
        sx, sy = get_smooth_xy(x_data, y_data[_])
        if broken_y_axis is False:
            if _ == 2:
                ax.fill_between(sx, sminy, smaxy,
                                color=color[_], edgecolor='none', alpha=0.2)
            plt.plot(sx, sy, label=label[_], color=color[_],
                     linestyle=linestyle[_], linewidth=3,
                     marker=marker[_], markevery=[],
                     markerfacecolor=color[_], markeredgecolor='none', markersize=8)
            # plt.plot(x_data, y_data[_], label=label[_],
            #          color=color[_], linestyle=linestyle[_], linewidth=3,
            #          marker=marker[_], markevery=[],
            #          markerfacecolor=color[_], markeredgecolor='none')
            # 'none'表示无填充颜色
            plt.scatter(x_data, y_data[_], color=color[_],
                        edgecolors='none', marker=marker[_], s=50)
        else:
            ax_top.plot(sx, sy, label=label[_],
                        color=color[_], linestyle=linestyle[_], linewidth=3,
                        marker=marker[_], markevery=[],
                        markerfacecolor=color[_], markeredgecolor='none', markersize=8)
            ax_bottom.plot(sx, sy, label=label[_],
                           color=color[_], linestyle=linestyle[_], linewidth=3,
                           marker=marker[_], markevery=[],
                           markerfacecolor=color[_], markeredgecolor='none', markersize=8)
            ax_top.scatter(x_data, y_data[_], color=color[_],
                           edgecolors='none', marker=marker[_], s=50)
            ax_bottom.scatter(x_data, y_data[_], color=color[_],
                              edgecolors='none', marker=marker[_], s=50)
    if broken_y_axis is True:
        # 绘制断裂处的斜线段
        kwargs = dict(color='k', linewidth=1, clip_on=False)
        xlim = ax_top.get_xlim()
        ylim_min = (ax_bottom.get_ylim())[0]
        ylim_max = (ax_top.get_ylim())[1]
        # 斜线段宽度占x轴的比例
        dx = 0.01*(xlim[1]-xlim[0])
        # 斜线段高度占y轴top的比例
        top_dy = 0.01*(ylim_max-ylim_min)
        ax_top.plot((xlim[0]-dx, xlim[0]+dx),
                    (ylim_top[0]-top_dy, ylim_top[0]+top_dy), **kwargs)
        ax_top.plot((xlim[1]-dx, xlim[1]+dx),
                    (ylim_top[0]-top_dy, ylim_top[0]+top_dy), **kwargs)
        # 斜线段高度占y轴bottom的比例
        bottom_dy = 0.01*(ylim_max-ylim_min)
        ax_bottom.plot((xlim[0]-dx, xlim[0]+dx),
                       (ylim_bottom[1] - bottom_dy,
                        ylim_bottom[1]+bottom_dy),
                       **kwargs)
        ax_bottom.plot((xlim[1]-dx, xlim[1]+dx),
                       (ylim_bottom[1] - bottom_dy,
                        ylim_bottom[1]+bottom_dy),
                       **kwargs)
        ax_top.set_xlim(xlim)
        ax_bottom.set_xlim(xlim)
    if broken_y_axis is False:
        # 设置网格线
        # plt.grid(axis='y', color='gray', linestyle='-')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.legend(loc=9, prop=font1)
        fig = plt.gcf()
    else:
        ax_top.grid(axis='y', color='gray', linestyle='-')
        ax_bottom.grid(axis='y', color='gray', linestyle='-')
        ax_top.legend(loc=9, prop=font1)
    # 设置图片的长和宽（英寸）
    fig.set_size_inches(9, 3.5)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print('已将图像保存为'+save_path)
    plt.show()
    if latex is True:
        mpl.rc('text', usetex=False)


def plot_zoom(x_data, y_data, save_path, x_label, y_label,
              var_orient='horizon'):
    """
    绘制缩放图
    """
    if var_orient != 'vertical' and var_orient != 'horizon':
        print('plot_figure var_orient error!')
        exit(1)
    if var_orient == 'vertical':
        y_data = list(map(list, zip(*y_data)))
    x_number = len(y_data[0])
    y_number = len(y_data)
    if len(x_data) != x_number:
        print('x_data和y_data不匹配！')
        exit(1)
    # 设置y_data的label、color、linestyle、marker
    if y_number == 1:
        label = ['']
    elif y_number == 2:
        label = ['label1', 'label2']
    elif y_number == 3:
        label = ['label1', 'label2', 'label3']
    else:
        label = []
    # HTML颜色名 'red', 'green', 'blue', 'yellow', 'black', 'white',
    # 'cyan', 'darksalmon', 'gold', 'crimson'
    if y_number == 1:
        color = ['red']
    elif y_number == 2:
        color = ['red', 'green']
    elif y_number == 3:
        color = ['red', 'green', 'black']
    else:
        color = []
    # 线条风格 '-', '--', '-.', ':', 'None'
    if y_number == 1:
        linestyle = ['-']
    elif y_number == 2:
        linestyle = ['-', '--']
    elif y_number == 3:
        linestyle = ['-', '--', '-.']
    else:
        linestyle = []
    # 线条标记 'o', 'd', 'D', 'h', 'H', '_', '8', 'p', ',', '.', 's',
    #  '+', '*', 'x', '^', 'v', '<', '>', '|'
    if y_number == 1:
        marker = ['o']
    elif y_number == 2:
        marker = ['v', '^']
    elif y_number == 3:
        marker = ['v', '^', 'o']
    else:
        marker = []
    if y_number != len(label) or y_number != len(color) or \
            y_number != len(linestyle) or y_number != len(marker):
        print('y_data的label、color、linestyle、marker没有被正确设置！')
        exit(1)
    # 设置标题
    # plt.title('title')
    # 设置字体和大小
    plt.rc('text', usetex=False)
    plt.rc('font', family='Times New Roman', size=15)
    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}
    # 设置缩放图
    fig, ax = plt.subplots()
    # 设置x轴的主刻度
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    # 设置x轴的副刻度
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    # 设置y轴的主刻度
    ax.yaxis.set_major_locator(MultipleLocator(5))
    # 设置y轴的副刻度
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    # x左 y下 x长度 y长度
    sub_ax = ax.inset_axes([0.30, 0.55, 0.35, 0.35])
    # 设置x轴的刻度标签
    plt.xticks(np.arange(1, x_number+1, 0.5), rotation=0)
    # 设置y轴的刻度标签
    plt.yticks(rotation=15)
    # 设置x轴的标签
    plt.xlabel(x_label, fontweight='bold', size=15)
    # 设置x轴的范围
    # plt.xlim([0, 5])
    # 设置y轴的标签
    plt.ylabel(y_label, fontweight='bold', size=15)
    # 设置y轴的范围
    plt.ylim([0, 25])
    # 为子图设置x轴的范围
    sub_ax.set_xlim(2, 3)
    # 为子图设置y轴的范围
    sub_ax.set_ylim(6, 10)
    for _ in range(y_number):
        sx, sy = get_smooth_xy(x_data, y_data[_])
        ax.plot(sx, sy, label=label[_],
                color=color[_], linestyle=linestyle[_], linewidth=3,
                marker=marker[_], markevery=[],
                markerfacecolor=color[_], markeredgecolor='k', markersize=8)
        # ax.plot(x_data, y_data[_], label=label[_],
        #         color=color[_], linestyle=linestyle[_], linewidth=3,
        #         marker=marker[_], markevery=[],
        #         markerfacecolor=color[_], markeredgecolor='k')
        ax.scatter(x_data, y_data[_], color=color[_],
                   edgecolors='k', marker=marker[_], s=50)
        sub_ax.plot(sx, sy, label=label[_],
                    color=color[_], linestyle=linestyle[_], linewidth=3,
                    marker=marker[_], markevery=[],
                    markerfacecolor=color[_], markeredgecolor='k', markersize=8)
        # sub_ax.plot(x_data, y_data[_], label=label[_],
        #             color=color[_], linestyle=linestyle[_], linewidth=3,
        #             marker=marker[_], markevery=[],
        #             markerfacecolor=color[_], markeredgecolor='k')
        sub_ax.scatter(x_data, y_data[_], color=color[_],
                       edgecolors='k', marker=marker[_], s=50)
    # 插入缩放图
    ax.indicate_inset_zoom(sub_ax, ec='gray', fc='gray')
    sub_ax.set_title('Subgraph', fontsize=8, fontweight='bold')
    # 设置网格线
    plt.grid(axis='y', color='gray', linestyle='-')
    # 设置图例
    handles, labels = ax.get_legend_handles_labels()
    for i in range(len(labels)):
        if labels[i] == 'indicate_inset':
            labels.pop(i)
            handles.pop(i)
            break
    # plt.legend(loc=1, prop=font1)
    plt.legend(handles, labels, bbox_to_anchor=(0.0, 1.0, 1.0, 0.0),
               loc='lower left', ncol=3, mode="expand", borderaxespad=0.,
               prop=font1)
    fig = plt.gcf()
    # 设置图片的长和宽（英寸）
    fig.set_size_inches(9, 3.5)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print('已将图像保存为'+save_path)
    plt.show()


def plot_bar(x_data, y_data, save_path, x_labels, y_label,
             var_orient='horizon', show_text=False):
    """
    绘制柱状图
    """
    # show_text指是否显示柱状条的值
    if var_orient != 'vertical' and var_orient != 'horizon':
        print('plot_figure var_orient error!')
        exit(1)
    if var_orient == 'vertical':
        y_data = list(map(list, zip(*y_data)))
    x_number = len(y_data[0])
    y_number = len(y_data)
    if len(x_data) != x_number:
        print('x_data和y_data不匹配！')
        exit(1)
    # 设置数据的标准差
    std_err = [[0.51, 0.11, 0.75, 0.43, 0.98, 0.71, 0.24],
               [0.84, 0.76, 0.29, 0.28, 0.33, 0.28, 0.12],
               [0.91, 0.23, 0.12, 0.62, 0.26, 0.27, 0.31]]
    # 设置x_data对应的标签
    x_data_label = ['Aaaa', 'Bbbb', 'Cccc',
                    'Dddd', 'Eeee', 'Ffff', 'Gggg']
    # x_data_label = ['Aaaa1', 'Aaaa2', 'Aaaa3',
    #                 'Bbbb1', 'Bbbb2', 'Bbbb3',
    #                 'Cccc1', 'Cccc2', 'Cccc3',
    #                 'Dddd1', 'Dddd2', 'Dddd3',
    #                 'Eeee1', 'Eeee2', 'Eeee3',
    #                 'Ffff1', 'Ffff2', 'Ffff3',
    #                 'Gggg1', 'Gggg2', 'Gggg3']
    if len(x_data_label) % x_number != 0:
        print('x_data对应的标签没有被正确设置！')
        exit(1)
    # 设置y_data的label、color、hatch
    if y_number == 1:
        label = ['label1']
    elif y_number == 2:
        label = ['label1', 'label2']
    elif y_number == 3:
        label = ['label1', 'label2', 'label3']
    else:
        label = []
    # HTML颜色名 'red', 'green', 'blue', 'yellow', 'black', 'white', 'cyan',
    #  'darksalmon', 'gold', 'crimson'
    if y_number == 1:
        color = ['green']
    elif y_number == 2:
        color = ['green', 'red']
    elif y_number == 3:
        color = ['green', 'red', 'blue']
    else:
        color = []
    # 填充效果 '/', '|', '-', '+', 'x', 'o', 'O', '.', '*', None
    if y_number == 1:
        hatch = ['x']
    elif y_number == 2:
        hatch = ['x', '.']
    elif y_number == 3:
        hatch = ['x', '.', '|']
    else:
        hatch = []
    if y_number != len(label) or y_number != len(color) or \
            y_number != len(hatch) or y_number != len(std_err):
        print('y_data的label、color、hatch没有被正确设置！')
        exit(1)
    # 设置标题
    # plt.title('title')
    # 设置字体和大小
    plt.rc('text', usetex=False)
    plt.rc('font', family='Times New Roman', size=15)
    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}
    # 获取当前Axes对象
    ax = plt.gca()
    # 设置y轴的主刻度
    ax.yaxis.set_major_locator(MultipleLocator(5))
    # 设置y轴的副刻度
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    # 设置柱状体的宽度
    bar_width = 0.20
    # 设置柱状体间隔占宽度的比例
    interval_ratio = 0.20
    # 设置x轴的标签
    xlabel_nb = len(x_labels)
    if xlabel_nb == 1:
        plt.xlabel(x_labels[0], fontweight='bold', size=15)
    else:
        label_loc = [0.75, 3.25, 5.75]
        line_min = -0.50
        line_max = 7.00
        line_loc = [line_min, 1.75, 4.75, line_max]
        # 设置x轴的范围
        plt.xlim([line_min, line_max])
        if len(label_loc) != xlabel_nb or len(line_loc) != xlabel_nb+1:
            print('label_loc、line_loc和x_labels不匹配！')
            exit(1)
        for _ in range(xlabel_nb):
            plt.text(
                x=label_loc[_], y=-2.4, s=x_labels[_],
                ha='center', va='center',
                fontsize=15, color='k', fontweight='bold')
        plt.text(x=(x_data[-1]-0.5)/2, y=-4.0, s='xlabel',
                 ha='center', va='center',
                 fontsize=15, color='k', fontweight='bold')
        for _ in range(xlabel_nb+1):
            plt.plot([line_loc[_], line_loc[_]], [0, -3],
                     color='k', linewidth=1, clip_on=False)
            plt.plot([line_loc[_], line_loc[_]], [0, 50],
                     color='k', linewidth=1, linestyle='--')
    # 设置x轴的刻度标签
    # plt.xticks(np.arange(x_number)+bar_width * (interval_ratio+1)
    #            * (y_number/2-0.5), x_data, rotation=0)
    plt.xticks(np.arange(x_number)+bar_width * (interval_ratio+1)
               * (y_number/2-0.5), x_data_label, rotation=0)
    # 设置y轴的标签
    plt.ylabel(y_label, fontweight='bold', size=15)
    # 设置y轴的范围
    plt.ylim([0, 15])
    # 设置误差棒的参数
    error_attri = {"elinewidth": 2, "ecolor": "black", "capsize": 6}
    for _ in range(y_number):
        plt.bar(np.arange(x_number)+_*bar_width*(interval_ratio+1),
                y_data[_], label=label[_], color=color[_], edgecolor='k',
                hatch=hatch[_], width=bar_width,
                yerr=std_err[_], error_kw=error_attri)
        if show_text is True:
            for __ in range(x_number):
                plt.text(x=__+bar_width * (interval_ratio+1)*_,
                         y=y_data[_][__], s=str(y_data[_][__]),
                         ha='center', va='bottom',
                         fontsize=8, color='red',
                         fontweight='bold')
    # 添加数据标注
    # '-', '->', '-[', '|-|', '-|>', '<-', '<->', '<|-', '<|-|>'
    # 'fancy', 'simple', 'wedge'
    plt.annotate('20', xytext=(5.8, 13.0), xy=(5.3, 14.5),
                 arrowprops=dict(arrowstyle='-|>', color='k', alpha=1),
                 # 'round', 'square'
                 bbox=dict(boxstyle='square',
                           fc='white', ec='k', lw=1, alpha=1))
    # 设置水平线
    plt.axhline(9, linestyle='--', color='r')
    # 设置垂直线
    plt.axvline(0.5, linestyle='--', color='r')
    # 设置网格线
    plt.grid(axis='y', color='gray', linestyle='-')
    # plt.legend(loc=1, prop=font1)
    plt.legend(bbox_to_anchor=(0.0, 1.0, 1.0, 0.0), loc='lower left',
               ncol=3, mode="expand", borderaxespad=0., prop=font1)
    fig = plt.gcf()
    # 设置图片的长和宽（英寸）
    fig.set_size_inches(9, 3.5)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print('已将图像保存为'+save_path)
    print('若图像出现重叠现象，调小bar_width的值！')
    plt.show()


def plot_horizon_bar(x_data, y_data, save_path, x_label, y_label,
                     var_orient='horizon', show_text=False):
    """
    绘制水平柱状图
    """
    # show_text指是否显示柱状条的值
    if var_orient != 'vertical' and var_orient != 'horizon':
        print('plot_figure var_orient error!')
        exit(1)
    if var_orient == 'vertical':
        y_data = list(map(list, zip(*y_data)))
    x_number = len(y_data[0])
    y_number = len(y_data)
    if len(x_data) != x_number:
        print('x_data和y_data不匹配！')
        exit(1)
    # 设置x_data对应的标签
    x_data_label = ['Aaaa', 'Bbbb', 'Cccc',
                    'Dddd', 'Eeee', 'Ffff', 'Gggg']
    # x_data_label = ['Aaaa1', 'Aaaa2', 'Aaaa3',
    #                 'Bbbb1', 'Bbbb2', 'Bbbb3',
    #                 'Cccc1', 'Cccc2', 'Cccc3',
    #                 'Dddd1', 'Dddd2', 'Dddd3',
    #                 'Eeee1', 'Eeee2', 'Eeee3',
    #                 'Ffff1', 'Ffff2', 'Ffff3',
    #                 'Gggg1', 'Gggg2', 'Gggg3']
    if len(x_data_label) % x_number != 0:
        print('x_data对应的标签没有被正确设置！')
        exit(1)
    # 设置y_data的label、color、hatch
    if y_number == 1:
        label = ['label1']
    elif y_number == 2:
        label = ['label1', 'label2']
    elif y_number == 3:
        label = ['label1', 'label2', 'label3']
    else:
        label = []
    # HTML颜色名 'red', 'green', 'blue', 'yellow', 'black', 'white', 'cyan',
    #  'darksalmon', 'gold', 'crimson'
    if y_number == 1:
        color = ['green']
    elif y_number == 2:
        color = ['green', 'red']
    elif y_number == 3:
        color = ['green', 'red', 'blue']
    else:
        color = []
    # 填充效果 '/', '|', '-', '+', 'x', 'o', 'O', '.', '*', None
    if y_number == 1:
        hatch = ['x']
    elif y_number == 2:
        hatch = ['x', '.']
    elif y_number == 3:
        hatch = ['x', '.', '|']
    else:
        hatch = []
    if y_number != len(label) or y_number != len(color) or \
            y_number != len(hatch):
        print('y_data的label、color、hatch没有被正确设置！')
        exit(1)
    # 设置标题
    # plt.title('title')
    # 设置字体和大小
    plt.rc('text', usetex=False)
    plt.rc('font', family='Times New Roman', size=15)
    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}
    # 获取当前Axes对象
    ax = plt.gca()
    # 设置x轴的主刻度
    ax.xaxis.set_major_locator(MultipleLocator(5))
    # 设置x轴的副刻度
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    # 设置柱状体的宽度
    bar_width = 0.20
    # 设置柱状体间隔占宽度的比例
    interval_ratio = 0.20
    # 设置y轴的标签
    plt.ylabel(x_label, fontweight='bold', size=15)
    # 设置y轴的范围
    # plt.ylim([0, 5])
    # 设置y轴的刻度标签
    # plt.yticks(np.arange(x_number)+bar_width * (interval_ratio+1)
    #            * (y_number/2-0.5), x_data, rotation=0)
    plt.yticks(np.arange(x_number)+bar_width * (interval_ratio+1)
               * (y_number/2-0.5), x_data_label, rotation=0)
    # 设置x轴的标签
    plt.xlabel(y_label, fontweight='bold', size=15)
    # 设置x轴的范围
    plt.xlim([0, 25])
    for _ in range(y_number):
        plt.barh(
            np.arange(x_number)+_*bar_width*(interval_ratio+1),
            y_data[_], label=label[_], color=color[_], edgecolor='k',
            alpha=0.5, hatch=hatch[_], height=bar_width)
        if show_text is True:
            for __ in range(x_number):
                plt.text(x=y_data[_][__],
                         y=__+bar_width * (interval_ratio+1)*_,
                         s=str(y_data[_][__]),
                         ha='left', va='center',
                         fontsize=8, color='blue',
                         fontweight='bold')
    # 设置网格线
    plt.grid(axis='x')
    # plt.legend(loc=1, prop=font1)
    plt.legend(bbox_to_anchor=(0.0, 1.0, 1.0, 0.0), loc='lower left',
               ncol=3, mode="expand", borderaxespad=0., prop=font1)
    fig = plt.gcf()
    # 设置图片的长和宽（英寸）
    fig.set_size_inches(9, 3.5)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print('已将图像保存为'+save_path)
    print('若图像出现重叠现象，调小bar_width的值！')
    plt.show()


def plot_line_bar(x_data, y_data, save_path, x_label, y_label,
                  var_orient='horizon'):
    """
    绘制双轴折线柱状图
    """
    print('从y_data[0]到y_data[a]-->左轴y_data')
    print('从y_data[a+1]到y_data[b]-->右轴y_data')
    if var_orient != 'vertical' and var_orient != 'horizon':
        print('plot_figure var_orient error!')
        exit(1)
    if var_orient == 'vertical':
        y_data = list(map(list, zip(*y_data)))
    x_number = len(y_data[0])
    y_number = len(y_data)
    if y_number % 2 != 0:
        print('y_number应该是偶数！')
        exit(1)
    y_number = int(y_number/2)
    y_data_left = y_data[0:y_number]
    y_data_right = y_data[y_number:2*y_number]
    if len(x_data) != x_number:
        print('x_data和y_data不匹配！')
        exit(1)
    # 设置x_data对应的标签
    x_data_label = ['Aaaa', 'Bbbb', 'Cccc', 'Dddd', 'Eeee', 'Ffff', 'Gggg']
    if len(x_data_label) % x_number != 0:
        print('x_data对应的标签没有被正确设置！')
        exit(1)
    # 设置y_data的label、color、hatch
    label_left = ['left1', 'left2', 'left3', 'left4']
    label_right = ['right1', 'right2', 'right3', 'right4']
    if len(label_left) != len(label_right):
        print('label_left应该与label_right相等！')
        exit(1)
    # HTML颜色名 'red', 'green', 'blue', 'yellow', 'black', 'white',
    # 'cyan', 'darksalmon', 'gold', 'crimson'
    color_left = ['red', 'green', 'blue', 'gray']
    color_right = ['cyan', 'darksalmon', 'gold', 'crimson']
    if len(color_left) != len(color_right):
        print('color_right应该与color_right相等！')
        exit(1)
    # 填充效果 '//', '|', '-', '+', 'x', 'o', 'O', '.', '*', None
    hatch = [None, None, None, None]
    # 线条标记 'o', 'd', 'D', 'h', 'H', '_', '8', 'p', ',', '.', 's',
    #  '+', '*', 'x', '^', 'v', '<', '>', '|'
    marker = ['^', 'o', 'd', 'h']
    # 线条风格 '-', '--', '-.', ':', 'None'
    linestyle = ['-', '--', '-.', ':']
    if y_number != len(label_left) or y_number != len(color_left) or \
            y_number != len(hatch):
        print('y_data的label、color、hatch、marker、linestyle没有被正确设置！')
        exit(1)
    # 设置标题
    # plt.title('title')
    # 设置字体和大小
    plt.rc('text', usetex=False)
    plt.rc('font', family='Times New Roman', size=15)
    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}
    # 设置柱状体的宽度
    bar_width = 0.15
    # 设置柱状体间隔占宽度的比例
    interval_ratio = 0.20
    # 生成左y轴
    fig, ax_left = plt.subplots()
    # 获取当前Axes对象
    ax = plt.gca()
    # 设置y轴的主刻度
    ax.yaxis.set_major_locator(MultipleLocator(5))
    # 设置y轴的副刻度
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    # 设置x轴的刻度标签
    # plt.xticks(np.arange(x_number)+bar_width * (interval_ratio+1)
    #            * (y_number/2-0.5), x_data, rotation=0)
    plt.xticks(np.arange(x_number)+bar_width * (interval_ratio+1)
               * (y_number/2-0.5), x_data_label, rotation=30)
    # 生成右y轴
    ax_right = ax_left.twinx()

    # 设置x轴的标签
    ax_left.set_xlabel(x_label, family='Times New Roman',
                       fontweight='bold', fontsize=15)
    # 设置x轴的范围
    # plt.xlim([0, 5])
    # 设置y轴的标签
    ax_left.set_ylabel(y_label[0], family='Times New Roman',
                       fontweight='bold', fontsize=15)
    ax_right.set_ylabel(y_label[1], family='Times New Roman',
                        fontweight='bold', fontsize=15)
    # 设置y轴的范围
    ax_left.set_ylim([0, 20])
    ax_right.set_ylim([0, 35])
    for _ in range(y_number):
        # 左y轴为柱状图
        ax_left.bar(np.arange(x_number)+_*bar_width * (interval_ratio+1),
                    y_data_left[_], label=label_left[_], color=color_left[_],
                    edgecolor='k', hatch=hatch[_], width=bar_width)
        # 右y轴为折线图
        ax_right.plot(np.arange(x_number) +
                      (y_number-1)*bar_width * (interval_ratio+1)/2,
                      y_data_right[_], label=label_right[_],
                      color=color_right[_], linestyle=linestyle[_],
                      linewidth=2, marker=marker[_], markevery=[],
                      markerfacecolor='none', markeredgecolor=color_right[_], markersize=8)
        # 'none'表示无填充颜色
        ax_right.scatter(np.arange(x_number) +
                         (y_number-1)*bar_width * (interval_ratio+1)/2,
                         y_data_right[_], color='none',
                         edgecolors=color_right[_], marker=marker[_], s=80)
    plt.grid(axis='y', color='gray', linestyle='-')
    # plt.legend(loc=1)
    handles_left, labels_left = ax_left.get_legend_handles_labels()
    handles_right, labels_right = ax_right.get_legend_handles_labels()
    plt.legend(handles_left+handles_right, labels_left+labels_right,
               bbox_to_anchor=(0, 1, 1, 0), loc='lower left',
               ncol=4, mode="expand", borderaxespad=0., prop=font1)
    fig = plt.gcf()
    # 设置图片的长和宽（英寸）
    fig.set_size_inches(9, 3.5)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print('已将图像保存为'+save_path)
    print('若图像出现重叠现象，调小bar_width的值！')
    plt.show()


def plot_stacked_bar(x_data, y_data, save_path, x_label, y_label,
                     var_orient='horizon', bars_nb=1, show_prop=False):
    """
    绘制堆叠柱状图
    """
    # bars_nb指柱状条的数量
    # show_prop指是否显示百分比
    print('从y_data[0]到y_data[a]-->从柱状条1的下层到上层')
    print('从y_data[a+1]到y_data[b]-->从柱状条2的下层到上层')
    print('从y_data[b+1]到y_data[c]-->从柱状条3的下层到上层')
    print('从左到右-->从柱状条1到柱状条3')
    if var_orient != 'vertical' and var_orient != 'horizon':
        print('plot_figure var_orient error!')
        exit(1)
    if var_orient == 'vertical':
        y_data = list(map(list, zip(*y_data)))
    x_number = len(y_data[0])
    y_number = len(y_data)
    y_per_bar = int(y_number/bars_nb)
    print('bars_nb=%d' % (bars_nb))
    print('y_per_bar=%d' % (y_per_bar))
    if len(x_data) != x_number:
        print('x_data和y_data不匹配！')
        exit(1)
    # 计算堆叠后的_y_data
    _y_data = copy.deepcopy(y_data)
    for i in range(y_number):
        for j in range(x_number):
            if i % y_per_bar >= 1:
                _y_data[i][j] += _y_data[i-1][j]
    # 百分比保留几位小数
    decimal = 2
    if show_prop is True:
        # 计算堆叠柱状图的百分比
        prop_y_data = copy.deepcopy(y_data)
        for i in range(y_number):
            max_i = y_per_bar*int(i/y_per_bar)+y_per_bar-1
            for j in range(x_number):
                prop_y_data[i][j] = round(
                    y_data[i][j]/_y_data[max_i][j], decimal)
    # 设置x_data对应的标签
    x_data_label = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    if x_number != len(x_data_label):
        print('x_data对应的标签没有被正确设置！')
        exit(1)
    # 设置y_data的label、color、hatch
    if bars_nb == 1:
        label = ['label1', 'label2',  'label3']
    elif bars_nb == 2:
        label = ['label1-1', 'label1-2',  'label1-3',
                 'label2-1', 'label2-2',  'label2-3']
    elif bars_nb == 3:
        label = ['label1-1', 'label1-2',  'label1-3',
                 'label2-1', 'label2-2',  'label2-3',
                 'label3-1', 'label3-2',  'label3-3']
        # label = ['label1-1', 'label1-2',
        #          'label2-1', 'label2-2',
        #          'label3-1', 'label3-2']
    else:
        label = []
    # HTML颜色名 'red', 'green', 'blue', 'yellow', 'black', 'white',
    # 'cyan', 'darksalmon', 'gold', 'crimson'
    if bars_nb == 1:
        color = ['green', 'red',  'darksalmon']
    elif bars_nb == 2:
        color = ['green', 'red',  'darksalmon',
                 'blue', 'yellow',  'crimson']
    elif bars_nb == 3:
        color = ['green', 'red',  'darksalmon',
                 'blue', 'yellow',  'crimson',
                 'green', 'red',  'darksalmon']
        # color = ['green', 'red',
        #          'darksalmon', 'blue',
        #          'yellow',  'crimson']
    else:
        color = []
    # 填充效果 '/', '//', '|', '-', '+', 'x', 'o', 'O', '.', '*', None
    if bars_nb == 1:
        hatch = [None, None, None]
    elif bars_nb == 2:
        hatch = [None, None, None,
                 '||', '||', '||']
    elif bars_nb == 3:
        hatch = [None, None, None,
                 '||', '||', '||',
                 '//', '//', '//']
        # hatch = [None, None,
        #          '|| ', '||',
        #          '//', '//']
    else:
        hatch = []
    if y_number != len(label) or y_number != len(color) or \
            y_number != len(hatch):
        print('y_data的label、color、hatch没有被正确设置！')
        exit(1)
    # 设置标题
    # plt.title('title')
    # 设置字体和大小
    plt.rc('text', usetex=False)
    plt.rc('font', family='Times New Roman', size=15)
    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}
    # 获取当前Axes对象
    ax = plt.gca()
    # 设置y轴的主刻度
    ax.yaxis.set_major_locator(MultipleLocator(10))
    # 设置y轴的副刻度
    ax.yaxis.set_minor_locator(MultipleLocator(2))
    # 设置柱状体的宽度
    bar_width = 0.20
    # 设置柱状体间隔占宽度的比例
    interval_ratio = 0.20
    # 设置x轴的标签
    plt.xlabel(x_label, fontweight='bold', size=15)
    # 设置x轴的范围
    # plt.xlim([0, 5])
    # 设置x轴的刻度标签
    # plt.xticks(np.arange(x_number)+bar_width * (interval_ratio+1)
    #            * (bars_nb/2-0.5), x_data, rotation=0)
    plt.xticks(np.arange(x_number)+bar_width * (interval_ratio+1)
               * (bars_nb/2-0.5), x_data_label, rotation=0)
    # 设置y轴的标签
    plt.ylabel(y_label, fontweight='bold', size=15)
    # 设置y轴的范围
    # plt.ylim([0, 65])
    for i in range(bars_nb):
        max_y_per_bar = (i+1)*y_per_bar
        for j in range(y_per_bar):
            j = max_y_per_bar-j-1
            plt.bar(np.arange(x_number)+i*bar_width*(interval_ratio+1),
                    _y_data[j], label=label[j], color=color[j], edgecolor='k',
                    hatch=hatch[j], width=bar_width)
    if show_prop is True:
        # 添加堆叠柱状图的百分比
        for i in range(y_number):
            for j in range(x_number):
                plt.text(j+int(i/y_per_bar)*bar_width*(interval_ratio+1),
                         _y_data[i][j]-y_data[i][j]/2,
                         s=str(prop_y_data[i][j]),
                         ha='center', va='center',
                         fontsize=8, color='black',
                         fontweight='bold')
    # 设置网格线
    plt.grid(axis='y', color='gray', linestyle='-')
    # 添加数据标注
    plt.annotate('', xytext=(5.25, 41.0), xy=(5.25, 57.5),
                 arrowprops=dict(arrowstyle='<->', color='k', alpha=1))
    plt.plot([5.15, 5.40], [57, 57], color='k',
             linewidth=1, linestyle='-')
    plt.text(5.1, 49, s='15', ha='center', va='center',
             fontsize=12, color='black', fontweight='bold')
    # plt.legend(loc=1)
    # plt.legend(bbox_to_anchor=(0.0, 1.0, 1.0, 0.0), loc='lower left',
    #            ncol=3, mode="expand", borderaxespad=0., prop=font1)
    plt.legend(bbox_to_anchor=(1.0, 0.5), loc='center left',
               borderaxespad=0., prop=font1)
    fig = plt.gcf()
    # 设置图片的长和宽（英寸）
    fig.set_size_inches(9, 3.5)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print('已将图像保存为'+save_path)
    print('若图像出现重叠现象，调小bar_width的值！')
    plt.show()


def plot_pie(data, label, save_path, show_prop=False):
    """
    绘制饼状图
    """
    # show_prop指是否显示百分比
    number = len(data)
    if show_prop is True:
        autopct = '%1.1f%%'
    else:
        autopct = None
    # HTML颜色名 'red', 'green', 'blue', 'yellow', 'black', 'white',
    # 'cyan', 'darksalmon', 'gold', 'crimson'
    color = ['red', 'green', 'blue', 'yellow', 'cyan', 'darksalmon']
    # explode指每一块饼离开中心的距离
    explode = [0.0, 0.0, 0.1, 0.0, 0.0, 0.0]
    if number != len(label) or number != len(color) or number != len(explode):
        print('y_data的label、color、explode没有被正确设置！')
        exit(1)
    # 设置标题
    if os.name == 'nt':  # Windows
        font_path = "C:/Windows/Fonts/simsun.ttc"
    else:  # Unix-like (Linux, macOS, etc.)
        font_path = "/usr/share/fonts/truetype/wincorefonts/simsun.ttc"
    chinese = FontProperties(fname=font_path, size=15)
    # simsun.ttc
    # simhei.ttf
    # simkai.ttf
    plt.rc('text', usetex=False)
    plt.title("饼状图", font=chinese)
    # 设置字体和大小
    plt.rc('font', family='Times New Roman', size=15)
    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}

    plt.pie(x=data, labels=label, explode=explode,
            colors=color, autopct=autopct, textprops={
                'fontsize': 15, 'color': 'black'},
            shadow=True, startangle=0)
    # plt.legend(bbox_to_anchor=(0.0, 1.0, 1.0, 0.0), loc='lower left',
    #            ncol=3, mode="expand", borderaxespad=0., prop=font1)
    plt.legend(bbox_to_anchor=(1.15, 0.5), loc='center left',
               borderaxespad=0., prop=font1)
    fig = plt.gcf()
    # 设置图片的长和宽（英寸）
    fig.set_size_inches(5, 4)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print('已将图像保存为'+save_path)
    plt.show()


def plot_3D_line(x_data, data, save_path, x_label, y_label, z_label,
                 var_orient='horizon'):
    """
    绘制三维折线图
    """
    if var_orient != 'vertical' and var_orient != 'horizon':
        print('plot_figure var_orient error!')
        exit(1)
    if var_orient == 'vertical':
        data = list(map(list, zip(*data)))
    x_number = len(data[0])
    y_number = len(data)
    if len(x_data) != x_number:
        print('x_data和data不匹配！')
        exit(1)
    # 设置data的label、color、linestyle、marker
    if y_number == 1:
        label = ['label1']
    elif y_number == 2:
        label = ['label1', 'label2']
    elif y_number == 3:
        label = ['l1', 'l2', 'l3']
    else:
        label = []
    # HTML颜色名 'red', 'green', 'blue', 'yellow', 'black', 'white',
    # 'cyan', 'darksalmon', 'gold', 'crimson'
    if y_number == 1:
        color = ['red']
    elif y_number == 2:
        color = ['red', 'green']
    elif y_number == 3:
        color = ['red', 'green', 'black']
    else:
        color = []
    # 线条风格 '-', '--', '-.', ':', 'None'
    if y_number == 1:
        linestyle = ['-']
    elif y_number == 2:
        linestyle = ['-', '--']
    elif y_number == 3:
        linestyle = ['-', '--', '-.']
    else:
        linestyle = []
    # 线条标记 'o', 'd', 'D', 'h', 'H', '_', '8', 'p', ',', '.', 's',
    #  '+', '*', 'x', '^', 'v', '<', '>', '|'
    if y_number == 1:
        marker = ['o']
    elif y_number == 2:
        marker = ['v', '^']
    elif y_number == 3:
        marker = ['v', '^', 'o']
    else:
        marker = []
    if y_number != len(label) or y_number != len(color) or \
            y_number != len(linestyle) or y_number != len(marker):
        print('data的label、color、linestyle、marker没有被正确设置！')
        exit(1)
    # 设置标题
    # plt.title('title')
    # 设置字体和大小
    plt.rc('text', usetex=False)
    plt.rc('font', family='Times New Roman', size=15)
    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}
    # 获取新的Axes对象
    ax = plt.subplot(projection='3d')
    # 设置x轴的主刻度
    ax.xaxis.set_major_locator(MultipleLocator(1))
    # 设置y轴的主刻度
    ax.yaxis.set_major_locator(MultipleLocator(1))
    # 设置y轴的副刻度
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    # 设置z轴的主刻度
    ax.zaxis.set_major_locator(MultipleLocator(5))
    # 设置刻度标签的旋转度
    ax.tick_params(labelrotation=0)
    # 设置x轴的刻度标签
    ax.set_xticks(x_data)
    ax.set_xticklabels(x_data, rotation=0)
    # 设置y轴的刻度标签
    ax.set_yticks(np.arange(y_number)+1)
    ax.set_yticklabels(label, rotation=0)
    # 设置x轴的标签
    ax.set_xlabel(x_label, fontweight='bold', size=15)
    # 设置x轴的范围
    # ax.set_xlim([0, 8])
    # 设置y轴的标签
    ax.set_ylabel(y_label, fontweight='bold', size=15)
    # 设置y轴的范围
    # ax.set_ylim([0, 4])
    # 设置z轴的标签
    ax.set_zlabel(z_label, fontweight='bold', size=15)
    # 设置y轴的范围
    ax.set_zlim([0, 25])
    max_x = [0 for _ in range(y_number)]
    max_z = [0 for _ in range(y_number)]

    for _ in range(y_number):
        max_z[_] = np.max(data[_])
        for __ in range(x_number):
            if data[_][__] == max_z[_]:
                max_x[_] = x_data[__]
        max_z[_] = round(max_z[_], 2)
        y_list = [_+1 for __ in range(x_number)]
        # 'none'表示无填充颜色
        ax.plot(x_data, y_list, data[_], label=label[_],
                color=color[_], linestyle=linestyle[_], linewidth=3)
        ax.scatter(
            x_data, y_list, data[_], color=color[_], edgecolors=color[_],
            marker=marker[_], s=10, linewidths=3, alpha=1)
        #  若文字被挡住则进一步调大zorder的值
        ax.text(x=max_x[_], y=_+1, z=max_z[_]+3, s='%.2f' % max_z[_],
                horizontalalignment='center', verticalalignment='top',
                backgroundcolor='white', zorder=5,
                fontsize=10, color='k', fontweight='bold')
    # 设置网格线
    plt.grid(axis='both', color='gray', linestyle='-')
    # 设置图例
    # plt.legend(bbox_to_anchor=(0.0, 1.0, 1.0, 0.0), loc='lower left',
    #            ncol=1, mode="expand", borderaxespad=0., prop=font1)
    fig = plt.gcf()
    # 设置图片的长和宽（英寸）
    fig.set_size_inches(4, 4)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight',
                    bbox_extra_artists=[ax.zaxis.label])
        print('已将图像保存为'+save_path)
    plt.show()


def plot_3D_bar(x_data, data, save_path, x_label, y_label, z_label,
                var_orient='horizon'):
    """
    绘制三维柱状图
    """
    if var_orient != 'vertical' and var_orient != 'horizon':
        print('plot_figure var_orient error!')
        exit(1)
    if var_orient == 'vertical':
        data = list(map(list, zip(*data)))
    x_number = len(data[0])
    y_number = len(data)
    if len(x_data) != x_number:
        print('x_data和data不匹配！')
        exit(1)
    # 设置x_data对应的标签
    x_data_label = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    if len(x_data_label) % x_number != 0:
        print('x_data对应的标签没有被正确设置！')
        exit(1)
    # 设置y_data的label、color、hatch
    if y_number == 1:
        label = ['label1']
    elif y_number == 2:
        label = ['label1', 'label2']
    elif y_number == 3:
        label = ['l1', 'l2', 'l3']
    else:
        label = []
    # HTML颜色名 'red', 'green', 'blue', 'yellow', 'black', 'white', 'cyan',
    #  'darksalmon', 'gold', 'crimson'
    if y_number == 1:
        color = ['green']
    elif y_number == 2:
        color = ['green', 'red']
    elif y_number == 3:
        color = ['green', 'red', 'blue']
    else:
        color = []
    # 填充效果 '/', '|', '-', '+', 'x', 'o', 'O', '.', '*', ' '
    if y_number == 1:
        hatch = ['x']
    elif y_number == 2:
        hatch = ['x', '.']
    elif y_number == 3:
        hatch = ['x', '.', '|']
    else:
        hatch = []
    if y_number != len(label) or y_number != len(color) or \
            y_number != len(hatch):
        print('y_data的label、color、hatch没有被正确设置！')
        exit(1)
    # 设置标题
    # plt.title('title')
    # 设置字体和大小
    plt.rc('text', usetex=False)
    plt.rc('font', family='Times New Roman', size=15)
    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}
    # 柱状体在x轴方向上的宽度
    bar_width_x = 0.5
    dx = [bar_width_x for _ in range(x_number)]
    # 柱状体在y轴方向上的宽度
    bar_width_y = 0.2
    dy = [bar_width_y for _ in range(x_number)]
    # 获取新的Axes对象
    ax = plt.subplot(projection='3d')
    # 设置x轴的主刻度
    ax.xaxis.set_major_locator(MultipleLocator(1))
    # 设置y轴的主刻度
    ax.yaxis.set_major_locator(MultipleLocator(1))
    # 设置y轴的副刻度
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    # 设置z轴的主刻度
    ax.zaxis.set_major_locator(MultipleLocator(5))
    # 设置刻度标签的旋转度
    ax.tick_params(labelrotation=0)
    # 设置x轴的刻度标签
    ax.set_xticks(x_data)
    ax.set_xticklabels(x_data_label, rotation=0)
    # 设置y轴的刻度标签
    ax.set_yticks(np.arange(y_number)+1)
    ax.set_yticklabels(label, rotation=0)
    # 设置x轴的标签
    ax.set_xlabel(x_label, fontweight='bold', size=15)
    # 设置x轴的范围
    # ax.set_xlim([0, 8])
    # 设置y轴的标签
    ax.set_ylabel(y_label, fontweight='bold', size=15)
    # 设置y轴的范围
    # ax.set_ylim([0, 4])
    # 设置z轴的标签
    ax.set_zlabel(z_label, fontweight='bold', size=15)
    # 设置y轴的范围
    ax.set_zlim([0, 25])
    min_x = [0 for _ in range(y_number)]
    min_z = [0 for _ in range(y_number)]
    for _ in range(y_number):
        min_z[_] = np.min(data[_])
        for __ in range(x_number):
            if data[_][__] == min_z[_]:
                min_x[_] = x_data[__]
        min_z[_] = round(min_z[_], 2)
        y_list = [_+1-1/2*bar_width_y for __ in range(x_number)]
        z_list = [0 for __ in range(x_number)]
        # 'none'表示无填充颜色
        ax.bar3d(x_data, y_list, z_list,
                 dx=dx, dy=dy, dz=data[_], label=label[_],
                 color=color[_], edgecolor='k', hatch=hatch[_])
        #  若文字被挡住则进一步调大zorder的值
        ax.text(x=min_x[_], y=_+1, z=min_z[_]+3, s='%.2f' % min_z[_],
                horizontalalignment='center', verticalalignment='top',
                backgroundcolor='white', zorder=5,
                fontsize=10, color='k', fontweight='bold')
    # 设置网格线
    plt.grid(axis='both', color='gray', linestyle='-')
    fig = plt.gcf()
    # 设置图片的长和宽（英寸）
    fig.set_size_inches(4, 4)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight',
                    bbox_extra_artists=[ax.zaxis.label])
        print('已将图像保存为'+save_path)
    plt.show()


def plot_3D_surface(x_data, y_data, data, save_path, x_label, y_label, z_label,
                    var_orient='horizon'):
    """
    绘制三维曲面图
    """
    if var_orient != 'vertical' and var_orient != 'horizon':
        print('plot_figure var_orient error!')
        exit(1)
    if var_orient == 'vertical':
        data = list(map(list, zip(*data)))
    x_number = len(data[0])
    y_number = len(data)
    if len(x_data) != x_number:
        print('x_data和data不匹配！')
        exit(1)
    # 设置x_data对应的标签
    x_data_label = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    if len(x_data_label) % x_number != 0:
        print('x_data对应的标签没有被正确设置！')
        exit(1)
    # 设置y_data的label、color、hatch
    if y_number == 1:
        label = ['label1']
    elif y_number == 2:
        label = ['label1', 'label2']
    elif y_number == 3:
        label = ['l1', 'l2', 'l3']
    else:
        label = []
    # HTML颜色名 'red', 'green', 'blue', 'yellow', 'black', 'white', 'cyan',
    #  'darksalmon', 'gold', 'crimson'
    if y_number == 1:
        color = ['green']
    elif y_number == 2:
        color = ['green', 'red']
    elif y_number == 3:
        color = ['green', 'red', 'blue']
    else:
        color = []
    # 填充效果 '/', '|', '-', '+', 'x', 'o', 'O', '.', '*', ' '
    if y_number == 1:
        hatch = ['x']
    elif y_number == 2:
        hatch = ['x', '.']
    elif y_number == 3:
        hatch = ['x', '.', '|']
    else:
        hatch = []
    if y_number != len(label) or y_number != len(color) or \
            y_number != len(hatch):
        print('y_data的label、color、hatch没有被正确设置！')
        exit(1)
    # 设置标题
    # plt.title('title')
    # 设置字体和大小
    plt.rc('text', usetex=False)
    plt.rc('font', family='Times New Roman', size=15)
    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}
    # 柱状体在x轴方向上的宽度
    bar_width_x = 0.5
    dx = [bar_width_x for _ in range(x_number)]
    # 柱状体在y轴方向上的宽度
    bar_width_y = 0.2
    dy = [bar_width_y for _ in range(x_number)]
    # 获取新的Axes对象
    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    # ax = plt.subplot(projection='3d')
    # 设置x轴的主刻度
    ax.xaxis.set_major_locator(MultipleLocator(1))
    # 设置y轴的主刻度
    ax.yaxis.set_major_locator(MultipleLocator(1))
    # 设置y轴的副刻度
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))
    # 设置z轴的主刻度
    ax.zaxis.set_major_locator(MultipleLocator(5))
    # 设置刻度标签的旋转度
    ax.tick_params(labelrotation=0)
    # 设置x轴的刻度标签
    ax.set_xticks(x_data)
    ax.set_xticklabels(x_data_label, rotation=0)
    # 设置y轴的刻度标签
    ax.set_yticks(y_data)
    ax.set_yticklabels(label, rotation=0)
    # 设置x轴的标签
    ax.set_xlabel(x_label, fontweight='bold', size=15)
    # 设置x轴的范围
    # ax.set_xlim([0, 8])
    # 设置y轴的标签
    ax.set_ylabel(y_label, fontweight='bold', size=15)
    # 设置y轴的范围
    # ax.set_ylim([0, 4])
    # 设置z轴的标签
    ax.set_zlabel(z_label, fontweight='bold', size=15)
    # 设置y轴的范围
    ax.set_zlim([0, 25])
    x = []
    y = []
    z = []
    for i in range(x_number):
        for j in range(y_number):
            x.append(x_data[i])
            y.append(y_data[j])
            z.append(data[j][i])
    surf = ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.1)
    min_x = [0 for _ in range(y_number)]
    min_z = [0 for _ in range(y_number)]
    for _ in range(y_number):
        min_z[_] = np.min(data[_])
        for __ in range(x_number):
            if data[_][__] == min_z[_]:
                min_x[_] = x_data[__]
        min_z[_] = round(min_z[_], 2)
        #  若文字被挡住则进一步调大zorder的值
        ax.text(x=min_x[_], y=y_data[_], z=min_z[_]+3, s='%.2f' % min_z[_],
                horizontalalignment='center', verticalalignment='top',
                backgroundcolor='white', zorder=5,
                fontsize=10, color='k', fontweight='bold')
    # 设置网格线
    plt.grid(axis='both', color='gray', linestyle='-')
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # 设置图片的长和宽（英寸）
    fig.set_size_inches(4, 4)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight',
                    bbox_extra_artists=[ax.zaxis.label])
        print('已将图像保存为'+save_path)
    plt.show()


def plot_confusion_matrix(data, save_path, x_label, y_label, is_norm=False):
    """
    绘制混淆矩阵
    """
    # is_norm指是否对数据进行正规化
    x_number = len(data[0])
    y_number = len(data)
    # 数据正规化
    if is_norm is True:
        norm_data = []
        for i in data:
            temp_data = []
            for j in i:
                temp_data.append(float(j/sum(i)))
            norm_data.append(temp_data)
    # 设置标题
    # plt.title('title')
    # 设置字体和大小
    plt.rc('text', usetex=False)
    plt.rc('font', family='Times New Roman', size=15)
    fig = plt.figure()
    ax = fig.add_subplot()
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    # 设置x轴的刻度标签
    plt.xticks(range(x_number), range(x_number), rotation=0)
    plt.xticks(range(x_number), alphabet[:x_number], rotation=0)
    # 设置y轴的刻度标签
    plt.yticks(range(y_number), range(y_number), rotation=0)
    plt.yticks(range(y_number), alphabet[:y_number], rotation=0)
    # 设置x轴的标签
    plt.xlabel(x_label, fontweight='bold', size=15)
    # 设置y轴的标签
    plt.ylabel(y_label, fontweight='bold', size=15)
    # jet, hot, gray, gist_yarg, Greys, Blues, Greens, Reds, Oranges, Purples
    if is_norm is True:
        res = ax.imshow(norm_data, cmap=plt.cm.jet, interpolation='nearest')
    else:
        res = ax.imshow(data, cmap=plt.cm.jet, interpolation='nearest')
    # 标注数据
    for x in range(x_number):
        for y in range(y_number):
            ax.annotate(str(data[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='k', fontsize=10, fontweight='normal')
    # 显示颜色条
    fig.colorbar(res)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print('已将图像保存为'+save_path)
    plt.show()


def piecewise_function(x):
    """
    定义分段函数
    """
    return np.piecewise(x, [x < 2.0, x >= 2.0],
                        [lambda x: x ** 2.0, lambda x: 4.0])


def plot_func(x_min, x_max, interval, save_path, x_label, y_label):
    """
    绘制数学函数图像
    y_data = [logn(2, x_data)]
    y_data = [logn(math.e, x_data)]
    y_data = [logn(10, x_data)]
    y_data = [power(2, x_data)]
    y_data = [power(math.e, x_data)]
    y_data = [sqrt(x_data)]
    y_data = [arcsin(x_data)]
    y_data = [arccos(x_data)]
    y_data = [arctanh(x_data)]
    """
    x_data = [np.linspace(x_min, x_max, interval),
              np.linspace(0, x_max, interval)]
    y_data = [piecewise_function(x_data[0]), logn(2, x_data[1])]
    y_number = len(y_data)
    # 设置y_data的label、color、linestyle、marker
    if y_number == 1:
        label = ['']
    elif y_number == 2:
        label = [r'$f(x)=\begin{cases}x^2&x<2\\4&x\geq2\end{cases}$',
                 r'$\log_{2}x$']
    elif y_number == 3:
        label = [r'$\log_{2}x$', r'$\log_{e}x$', r'$\log_{10}x$']
    elif y_number == 4:
        label = [r'$\log_{2}x$', r'$\log_{e}x$', r'$\log_{10}x$', r'$\lg x$']
    else:
        label = []
    # HTML颜色名 'red', 'green', 'blue', 'yellow', 'black', 'white',
    # 'cyan', 'darksalmon', 'gold', 'crimson'
    if y_number == 1:
        color = ['red']
    elif y_number == 2:
        color = ['red', 'green']
    elif y_number == 3:
        color = ['red', 'green', 'black']
    elif y_number == 4:
        color = ['red', 'green', 'black', 'blue']
    else:
        color = []
    # 线条风格 '-', '--', '-.', ':', 'None'
    if y_number == 1:
        linestyle = ['-']
    elif y_number == 2:
        linestyle = ['-', '--']
    elif y_number == 3:
        linestyle = ['-', '--', '-.']
    elif y_number == 4:
        linestyle = ['-', '--', '-.', ':']
    else:
        linestyle = []
    if y_number != len(label) or y_number != len(color) or \
            y_number != len(linestyle):
        print('y_data的label、color、linestyle没有被正确设置！')
        exit(1)
    # 设置标题
    # plt.title('title')
    # 设置字体和大小
    plt.rc('text', usetex=True)
    plt.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
    plt.rc('font', family='Times New Roman', size=15)
    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}
    # 获取当前Axes对象
    ax = plt.gca()
    ax.axis([x_data[0][0], x_data[0][-1], x_data[0][0], x_data[0][-1]])
    ax.spines['left'].set_position(('data', 0))
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['top'].set_color('none')
    # 设置x轴的主刻度
    ax.xaxis.set_major_locator(MultipleLocator(1))
    # 设置x轴的副刻度
    # ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    # 设置x轴的刻度标签
    plt.xticks(np.arange(int(x_min), int(x_max)+1, 1), rotation=0)
    # 设置y轴的主刻度
    ax.yaxis.set_major_locator(MultipleLocator(1))
    # 设置y轴的副刻度
    # ax.yaxis.set_minor_locator(MultipleLocator(0.2))
    # 设置x轴的刻度标签
    plt.xticks(rotation=0)
    # 设置y轴的刻度标签
    plt.yticks(rotation=0)
    # 设置x轴的标签
    plt.xlabel(x_label, fontweight='bold', size=15)
    # 设置x轴的范围
    # plt.xlim([-2, 8])
    # 设置y轴的标签
    plt.ylabel(y_label, fontweight='bold', size=15)
    # 设置y轴的范围
    plt.ylim([-1, 5])
    for _ in range(y_number):
        plt.plot(x_data[_], y_data[_], label=label[_],
                 color=color[_], linestyle=linestyle[_], linewidth=2)
    # 设置网格线
    plt.grid(axis='both', color='gray', linestyle='--')
    # plt.legend(loc=2, prop=font1)
    plt.legend(bbox_to_anchor=(0.0, 1.0, 1.0, 0.0), loc='lower left',
               ncol=4, mode="expand", borderaxespad=0., prop=font1)
    fig = plt.gcf()
    # 设置图片的长和宽（英寸）
    fig.set_size_inches(6.5, 3.5)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print('已将图像保存为'+save_path)
    plt.show()


def plot_fitting_curve(x_min, x_max, interval, x_point, y_point,
                       save_path, x_label, y_label):
    """
    绘制拟合曲线图像（曲线和散点图）
    y_data = [logn(2, x_data)]
    y_data = [logn(math.e, x_data)]
    y_data = [logn(10, x_data)]
    y_data = [power(2, x_data)]
    y_data = [power(math.e, x_data)]
    y_data = [sqrt(x_data)]
    y_data = [arcsin(x_data)]
    y_data = [arccos(x_data)]
    y_data = [arctanh(x_data)]
    """
    x_data = np.linspace(x_min, x_max, interval)
    y_data = [logn(2, x_data), y_point]
    y_number = len(y_data)
    # 设置y_data的label、color、linestyle、marker
    if y_number == 1:
        label = ['']
    elif y_number == 2:
        label = [r'$\log_{2}x$', "Sample Data"]
    elif y_number == 3:
        label = ['label1', 'label2', 'label3']
    elif y_number == 4:
        label = ['label1', 'label2', 'label3', 'label4']
    else:
        label = []
    # HTML颜色名 'red', 'green', 'blue', 'yellow', 'black', 'white',
    # 'cyan', 'darksalmon', 'gold', 'crimson'
    if y_number == 1:
        color = ['red']
    elif y_number == 2:
        color = ['red', 'green']
    elif y_number == 3:
        color = ['red', 'green', 'black']
    elif y_number == 4:
        color = ['red', 'green', 'black', 'blue']
    else:
        color = []
    # 线条风格 '-', '--', '-.', ':', 'None'
    if y_number == 1:
        linestyle = ['-']
    elif y_number == 2:
        linestyle = ['-', '--']
    elif y_number == 3:
        linestyle = ['-', '--', '-.']
    elif y_number == 4:
        linestyle = ['-', '--', '-.', ':']
    else:
        linestyle = []
    # 线条标记 'o', 'd', 'D', 'h', 'H', '_', '8', 'p', ',', '.', 's',
    #  '+', '*', 'x', '^', 'v', '<', '>', '|'
    if y_number == 1:
        marker = ['o']
    elif y_number == 2:
        marker = ['v', '^']
    elif y_number == 3:
        marker = ['v', '^', 'o']
    else:
        marker = []
    if y_number != len(label) or y_number != len(color) or \
            y_number != len(linestyle) or y_number != len(marker):
        print('y_data的label、color、linestyle没有被正确设置！')
        exit(1)
    # 设置标题
    # plt.title('title')
    # 设置字体和大小
    plt.rc('text', usetex=True)
    plt.rc('font', family='Times New Roman', size=15)
    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}
    # 获取当前Axes对象
    ax = plt.gca()
    # 设置x轴的主刻度
    ax.xaxis.set_major_locator(MultipleLocator(1))
    # 设置x轴的副刻度
    # ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    # 设置x轴的刻度标签
    plt.xticks(np.arange(int(x_min), int(x_max)+1, 1), rotation=0)
    # 设置y轴的主刻度
    ax.yaxis.set_major_locator(MultipleLocator(1))
    # 设置y轴的副刻度
    # ax.yaxis.set_minor_locator(MultipleLocator(0.2))
    # 设置x轴的刻度标签
    plt.xticks(rotation=0)
    # 设置y轴的刻度标签
    plt.yticks(rotation=0)
    # 设置x轴的标签
    plt.xlabel(x_label, fontweight='bold', size=15)
    # 设置x轴的范围
    # plt.xlim([0, 5])
    # 设置y轴的标签
    plt.ylabel(y_label, fontweight='bold', size=15)
    # 设置y轴的范围
    plt.ylim([-1, 5])
    for _ in range(y_number):
        if _ == 0:
            plt.plot(x_data, y_data[_], label=label[_],
                     color=color[_], linestyle=linestyle[_], linewidth=2)
        else:
            plt.scatter(x_point, y_data[_], color=color[_], label=label[_],
                        edgecolors='none', marker=marker[_], s=50)
    # 设置网格线
    plt.grid(axis='both', color='gray', linestyle='--')
    plt.legend(loc=2, prop=font1)
    # plt.legend(bbox_to_anchor=(0.0, 1.0, 1.0, 0.0), loc='lower left',
    #            ncol=4, mode="expand", borderaxespad=0., prop=font1)
    fig = plt.gcf()
    # 设置图片的长和宽（英寸）
    fig.set_size_inches(6.5, 3.5)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print('已将图像保存为'+save_path)
    plt.show()


def plot_actual_vs_predicted_with_histograms(X_train, y_train_pred, X_test, y_test_pred, save_path):
    train_r2 = r2_score(X_train, y_train_pred)
    test_r2 = r2_score(X_test, y_test_pred)

    plt.rc('text', usetex=True)
    mpl.rc('text.latex', preamble=r'\usepackage{amsmath}')
    plt.rc('font', family='Times New Roman', size=15)
    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 13}

    g = sns.jointplot()
    g.ax_joint.scatter(X_train, y_train_pred, color='blue',
                       label=r'\textbf{Train $\boldsymbol{R^2}$: }'+f'{train_r2:.2f}')
    g.ax_joint.scatter(X_test, y_test_pred, color='red',
                       label=r'\textbf{Test $\boldsymbol{R^2}$: }'+f'{test_r2:.2f}')

    max_val = max(X_train.max(), X_test.max(),
                  y_train_pred.max(), y_test_pred.max())
    g.ax_joint.plot([0, max_val], [0, max_val], '--', color='black')
    sns.regplot(x=X_test, y=y_test_pred, ax=g.ax_joint,
                scatter=False, color='red', label='')

    g.ax_joint.set_xlabel(r'\textbf{Actual Value}')
    g.ax_joint.set_ylabel(r'\textbf{Predicted Value}')
    g.ax_joint.legend(loc='upper left', prop=font1)

    sns.histplot(X_train, ax=g.ax_marg_x, color='blue', kde=True, alpha=0.5)
    sns.histplot(X_test, ax=g.ax_marg_x, color='red', kde=True, alpha=0.5)
    sns.histplot(y=y_train_pred, ax=g.ax_marg_y, color='blue',
                 kde=True, orientation='horizontal', alpha=0.5)
    sns.histplot(y=y_test_pred, ax=g.ax_marg_y, color='red',
                 kde=True, orientation='horizontal', alpha=0.5)

    fig = plt.gcf()
    fig.set_size_inches(4, 4)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print('已将图像保存为'+save_path)
    plt.show()


def plot_box_fig(save_path, *data_sets, labels=None):
    """
    绘制箱状图
    """
    plt.rc('text', usetex=False)
    plt.rc('font', family='Times New Roman', size=15)
    plt.xlabel('Type', fontweight='bold', size=15)
    plt.ylabel('Value', fontweight='bold', size=15)
    boxprops = dict(color='blue')
    medianprops = dict(color='green', linestyle='-', linewidth=2)
    meanlineprops = dict(color='red', linestyle='--', linewidth=2)
    plt.boxplot(data_sets, boxprops=boxprops, medianprops=medianprops,
                showmeans=True, meanline=True, meanprops=meanlineprops)
    if labels:
        plt.xticks(range(1, len(labels) + 1), labels)
    fig = plt.gcf()
    fig.set_size_inches(5, 4)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print('已将图像保存为'+save_path)
    plt.show()


def sample_figure():
    # cd python;rm -rf __pycache__;cd ..
    x_data = [1, 2, 3, 4, 5, 6, 7]
    y_data1 = [[3, 13, 1, 7, 9, 10, 2],
               [6, 3, 9, 4, 9, 20, 9],
               [7, 9, 5, 8, 5, 3, 9]]
    y_data2 = [[3, 13, 5, 7, 9, 10, 2],
               [6, 3, 8, 4, 9, 14, 9],
               [7, 9, 5, 8, 5, 3, 9],
               [8, 18, 10, 12, 14, 15, 7],
               [11, 8, 13, 9, 4, 19, 14],
               [12, 14, 10, 13, 10, 8, 14],
               [13, 23, 15, 17, 19, 20, 12],
               [16, 13, 18, 14, 9, 24, 19]]
    y_data3 = [[3, 13, 5, 7, 9, 10, 2],
               [6, 3, 8, 4, 9, 14, 9],
               [7, 9, 5, 8, 5, 3, 9],
               [8, 18, 10, 12, 14, 15, 7],
               [11, 8, 13, 9, 4, 19, 14],
               [12, 14, 10, 13, 10, 8, 14],
               [13, 23, 15, 17, 19, 20, 12],
               [16, 13, 18, 14, 9, 24, 19],
               [17, 19, 15, 18, 15, 13, 19]]
    y_data4 = [23, 56, 89, 15, 78, 43]
    data5 = [[1, 0.33, 0.48, 0.47, 0.4, 0.51, 0.48, 0.43, 0.5, 0.47],
             [0.33, 1, 0.38, 0.36, 0.36, 0.33, 0.34, 0.35, 0.39, 0.33],
             [0.48, 0.38, 1, 0.54, 0.5, 0.5, 0.49, 0.48, 0.55, 0.47],
             [0.47, 0.36, 0.54, 1, 0.42, 0.58, 0.42, 0.47, 0.54, 0.49],
             [0.4, 0.36, 0.54, 0.42, 1,  0.46, 0.49, 0.49, 0.51, 0.53],
             [0.51, 0.33, 0.5, 0.58, 0.46, 1, 0.5, 0.46, 0.6, 0.52],
             [0.48, 0.34, 0.49, 0.42, 0.49, 0.5, 1, 0.38, 0.5, 0.41],
             [0.43, 0.35, 0.48, 0.47, 0.49, 0.46, 0.38, 1, 0.49, 0.55],
             [0.5, 0.39, 0.55, 0.54, 0.51, 0.6, 0.5, 0.49, 1, 0.53],
             [0.47, 0.33, 0.47, 0.49, 0.53, 0.52, 0.41, 0.55, 0.53, 1]]
    x_point = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y_point = [0.2, 0.8, 1.5, 2.2, 2.3, 2.5, 3, 3.2, 3.3, 3.4]
    plot_scatter(x_data, y_data1, 'python/sample_figure/figure1.pdf',
                 'xlabel ($\mathbf{×10^{-3}}$)', 'ylabel', 'horizon')
    x_min = [2.5, 3, 9, 6, 4]
    x_max = [2.5, 4, 9.5, 9, 5.5]
    y_min = [85, 76, 83, 80, 90]
    y_max = [85, 81, 90, 85, 96]
    plot_ellipse(x_min, x_max, y_min, y_max, 'python/sample_figure/figure2.pdf',
                 'xlabel', 'ylabel')
    plot_line(x_data, y_data1, 'python/sample_figure/figure3-1.pdf',
              r'\textbf{xlabel}', r'\textbf{ylabel}', 'horizon')
    plot_line(x_data, y_data1, 'python/sample_figure/figure3-2.pdf',
              r'\textbf{xlabel}', r'\textbf{ylabel}', 'horizon', True)
    plot_zoom(x_data, y_data1, 'python/sample_figure/figure4.pdf',
              'xlabel', 'ylabel', 'horizon')
    plot_bar(x_data, y_data1, 'python/sample_figure/figure5.pdf',
             ['xlabel1', 'xlabel2', 'xlabel3'], 'ylabel', 'horizon')
    plot_horizon_bar(x_data, y_data1, 'python/sample_figure/figure6.pdf',
                     'xlabel', 'ylabel', 'horizon', True)
    plot_line_bar(x_data, y_data2, 'python/sample_figure/figure7.pdf',
                  'xlabel', ['ylabel left', 'ylabel right'], 'horizon')
    plot_stacked_bar(x_data, y_data3, 'python/sample_figure/figure8.pdf',
                     'xlabel', 'ylabel', 'horizon', 3)
    plot_pie(y_data4, ['label1', 'label2', 'label3',
                       'label4', 'label5', 'label6'],
             'python/sample_figure/figure9.pdf', True)
    plot_3D_line(x_data, y_data1, 'python/sample_figure/figure10.pdf',
                 '\nxlabel', '\nylabel', '\nzlabel', 'horizon')
    plot_3D_bar(x_data, y_data1, 'python/sample_figure/figure11.pdf',
                '\nxlabel', '\nylabel', '\nzlabel', 'horizon')
    plot_3D_surface(
        x_data, [1, 2, 3], y_data1, 'python/sample_figure/figure12.pdf',
        '\nxlabel', '\nylabel', '\nzlabel', 'horizon')
    plot_confusion_matrix(data5, 'python/sample_figure/figure13.pdf',
                          'Digit', 'Digit')
    plot_func(-2, 8, 200, 'python/sample_figure/figure14.pdf',
              r'\textbf{x}', r'\textbf{y}')
    plot_fitting_curve(0, 10, 200, x_point, y_point,
                       'python/sample_figure/figure15.pdf',
                       r'\textbf{x}', r'\textbf{y}')
    num_train = 50
    num_test = 30
    X_train = np.random.rand(num_train) * 80
    y_train_pred = X_train + np.random.randn(num_train) * 10
    X_test = np.random.rand(num_test) * 80
    y_test_pred = X_test + np.random.randn(num_test) * 10
    plot_actual_vs_predicted_with_histograms(
        X_train, y_train_pred, X_test, y_test_pred,
        'python/sample_figure/figure16.pdf')
    np.random.seed(10)
    data1 = np.random.normal(0, 1, 100)
    data2 = np.random.normal(1, 2, 100)
    data3 = np.random.normal(2, 1.5, 100)
    data4 = np.random.normal(3, 2, 100)
    plot_box_fig('python/sample_figure/figure17.pdf', data1, data2,
                 data3, data4, labels=['data1', 'data2', 'data3', 'data4'])


if __name__ == '__main__':
    # 在 Ubuntu 系统下运行以下脚本：
    # ~/shell/python.sh
    # ~/shell/latex.sh 1
    # ~/shell/latex.sh 2
    # 在 Windows 系统下：
    # 1. 下载并安装 Python：https://www.python.org/downloads/
    # 2. 以管理员模式打开 cmd
    # 3. 运行以下命令安装依赖：
    #    pip install matplotlib scipy seaborn scikit-learn PyPDF2
    # 4. 下载并安装 MiKTeX：https://miktex.org/download
    # ATC/NSDI顶会图像字体：https://www.cufonfonts.com/font/sf-pro-display
    sample_figure()
