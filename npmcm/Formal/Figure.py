#!/usr/bin/python3
# -*- coding:utf-8 -*-
import math
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as scinpo
import xlrd
from matplotlib.ticker import MultipleLocator
from numpy.lib.scimath import logn, power, sqrt

import data_process as dp

smiles_nb = 1974

pwd = os.getcwd()
res = pwd.split('/')
if res[-1] == "npmcm":
    pwd += "/Formal"
elif res[-1] == "reuse-code":
    pwd += "/npmcm/Formal"


def plot_func1(x_min, x_max, interval, save_path, x_label, y_label):
    """
    1/(1+e^(-x))
    """
    x_data = np.linspace(x_min, x_max, interval)
    y_data = [1/(1+power(math.e, -x_data)),
              -1*power(math.e, -x_data)/((1+power(math.e, -x_data))**2)]
    y_number = len(y_data)
    # 设置y_data的label、color、linestyle、marker
    label = [r'$Sigmoid(x)=\frac{1}{1+e^{-x}}$',
             r'$Sigmoid\'(x)=\frac{-e^{-x}}{(1+e^{-x})^2}$']
    # HTML颜色名 'red', 'green', 'blue', 'yellow', 'black', 'white',
    # 'cyan', 'darksalmon', 'gold', 'crimson'
    color = ['red', 'green']
    # 线条风格 '-', '--', '-.', ':', 'None'
    linestyle = ['-', '--']
    if y_number != len(label) or y_number != len(color) or \
            y_number != len(linestyle):
        print('y_data的label、color、linestyle没有被正确设置！')
        exit(1)
    # 设置字体和大小
    mpl.rcParams.update({'font.family': 'Times New Roman',
                         'mathtext.fontset': 'stix'})
    plt.rc('font', family='Times New Roman', size=15)
    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 14}
    # plt.title('title')
    # 获取当前Axes对象
    ax = plt.gca()
    ax.axis([x_data[0], x_data[-1], x_data[0], x_data[-1]])
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
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
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
    plt.ylim([-0.5, 2])
    for _ in range(y_number):
        plt.plot(x_data, y_data[_], label=label[_],
                 color=color[_], linestyle=linestyle[_], linewidth=2)
    # 设置网格线
    plt.grid(axis='both', color='gray', linestyle='--')
    plt.legend(loc=1, prop=font1)
    # plt.legend(bbox_to_anchor=(0.0, 1.0, 1.0, 0.0), loc='lower left',
    #            ncol=4, mode="expand", borderaxespad=0., prop=font1)
    fig = plt.gcf()
    # 设置图片的长和宽（英寸）
    fig.set_size_inches(7, 3.5)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print('已将图像保存为'+save_path)
    plt.show()


def piecewise_function(x):
    return np.piecewise(x, [x <= 0, x > 0],
                        [lambda x: 0, lambda x: x])


def piecewise_function_der(x):
    return np.piecewise(x, [x <= 0, x > 0],
                        [lambda x: 0, lambda x: 1])


def plot_func2(x_min, x_max, interval, save_path, x_label, y_label):
    """
    {0,x}
    """
    x_data = np.linspace(x_min, x_max, interval)
    y_data = [piecewise_function(x_data), piecewise_function_der(x_data)]
    y_number = len(y_data)
    # 设置y_data的label、color、linestyle、marker
    label = [r'$ReLu(x)$',
             r'$ReLu\'(x)$']
    # HTML颜色名 'red', 'green', 'blue', 'yellow', 'black', 'white',
    # 'cyan', 'darksalmon', 'gold', 'crimson'
    color = ['red', 'green']
    # 线条风格 '-', '--', '-.', ':', 'None'
    linestyle = ['-', '--']
    if y_number != len(label) or y_number != len(color) or \
            y_number != len(linestyle):
        print('y_data的label、color、linestyle没有被正确设置！')
        exit(1)
    # 设置字体和大小
    mpl.rcParams.update({'font.family': 'Times New Roman',
                         'mathtext.fontset': 'stix'})
    plt.rc('font', family='Times New Roman', size=15)
    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 14}
    # plt.title('title')
    # 获取当前Axes对象
    ax = plt.gca()
    ax.axis([x_data[0], x_data[-1], x_data[0], x_data[-1]])
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
    # plt.xlim([0, 5])
    # 设置y轴的标签
    plt.ylabel(y_label, fontweight='bold', size=15)
    # 设置y轴的范围
    plt.ylim([-1, 4])
    for _ in range(y_number):
        plt.plot(x_data, y_data[_], label=label[_],
                 color=color[_], linestyle=linestyle[_], linewidth=2)
    # 设置网格线
    plt.grid(axis='both', color='gray', linestyle='--')
    plt.legend(loc=2, prop=font1)
    # plt.legend(bbox_to_anchor=(0.0, 1.0, 1.0, 0.0), loc='lower left',
    #            ncol=4, mode="expand", borderaxespad=0., prop=font1)
    fig = plt.gcf()
    # 设置图片的长和宽（英寸）
    fig.set_size_inches(7.0, 3.5)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print('已将图像保存为'+save_path)
    plt.show()


def plot_func3(x_min, x_max, interval, save_path, x_label, y_label):
    """
    x/(1+e^(-βx))
    """
    x_data = np.linspace(x_min, x_max, interval)
    y_data = [x_data/(1+power(math.e, -x_data)),
              (1+power(math.e, -x_data)*(1+x_data)) /
              ((1+power(math.e, -x_data))**2)]
    y_number = len(y_data)
    # 设置y_data的label、color、linestyle、marker
    label = \
        [r'$Swish(x)=\frac{x}{1+e^{-\beta x}}$',
         r'$Swish\'(x)=\frac{1+(1+\beta x)e^{-\beta x}}{(1+e^{-\beta x})^2}$']
    # HTML颜色名 'red', 'green', 'blue', 'yellow', 'black', 'white',
    # 'cyan', 'darksalmon', 'gold', 'crimson'
    color = ['red', 'green']
    # 线条风格 '-', '--', '-.', ':', 'None'
    linestyle = ['-', '--']
    if y_number != len(label) or y_number != len(color) or \
            y_number != len(linestyle):
        print('y_data的label、color、linestyle没有被正确设置！')
        exit(1)
    # 设置字体和大小
    mpl.rcParams.update({'font.family': 'Times New Roman',
                         'mathtext.fontset': 'stix'})
    plt.rc('font', family='Times New Roman', size=15)
    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 14}
    # plt.title('title')
    # 获取当前Axes对象
    ax = plt.gca()
    ax.axis([x_data[0], x_data[-1], x_data[0], x_data[-1]])
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
    # plt.xlim([0, 5])
    # 设置y轴的标签
    plt.ylabel(y_label, fontweight='bold', size=15)
    # 设置y轴的范围
    plt.ylim([-1, 4])
    for _ in range(y_number):
        plt.plot(x_data, y_data[_], label=label[_],
                 color=color[_], linestyle=linestyle[_], linewidth=2)
    # 设置网格线
    plt.grid(axis='both', color='gray', linestyle='--')
    plt.legend(loc=2, prop=font1)
    # plt.legend(bbox_to_anchor=(0.0, 1.0, 1.0, 0.0), loc='lower left',
    #            ncol=4, mode="expand", borderaxespad=0., prop=font1)
    fig = plt.gcf()
    # 设置图片的长和宽（英寸）
    fig.set_size_inches(7.0, 3.5)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print('已将图像保存为'+save_path)
    plt.show()


def plot_fitting_curve(x_min, x_max, interval, x_point, y_point,
                       save_path, x_label, y_label):
    """
    绘制拟合曲线图像（曲线和散点图）
    """
    x_data = np.linspace(x_min, x_max, interval)
    y_data = \
        [75.75224+1.86394e6/(sqrt(2*math.pi)*0.25985*x_data) *
         power(math.e, -0.5*((logn(math.e, (x_data/2.38198))/0.25985)**2)),
         y_point]
    y_number = len(y_data)
    # 设置y_data的label、color、linestyle、marker
    label = ["Fitting Curve", "Sample Data"]
    # HTML颜色名 'red', 'green', 'blue', 'yellow', 'black', 'white',
    # 'cyan', 'darksalmon', 'gold', 'crimson'
    color = ['red', 'green']
    # 线条风格 '-', '--', '-.', ':', 'None'
    linestyle = ['-', '--']
    # 线条标记 'o', 'd', 'D', 'h', 'H', '_', '8', 'p', ',', '.', 's',
    #  '+', '*', 'x', '^', 'v', '<', '>', '|'
    marker = ['v', '^']
    if y_number != len(label) or y_number != len(color) or \
            y_number != len(linestyle) or y_number != len(marker):
        print('y_data的label、color、linestyle没有被正确设置！')
        exit(1)
    # 设置字体和大小
    mpl.rcParams.update({'font.family': 'Times New Roman',
                         'mathtext.fontset': 'stix'})
    plt.rc('font', family='Times New Roman', size=15)
    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}
    # plt.title('title')
    # 获取当前Axes对象
    ax = plt.gca()
    # 设置x轴的主刻度
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    # 设置x轴的副刻度
    # ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    # 设置x轴的刻度标签
    plt.xticks(np.arange(int(x_min), int(x_max)+1, 1), rotation=0)
    # 设置y轴的主刻度
    ax.yaxis.set_major_locator(MultipleLocator(3e5))
    # 设置y轴的副刻度
    # ax.yaxis.set_minor_locator(MultipleLocator(0.2))
    # 设置x轴的刻度标签
    plt.xticks(rotation=0)
    # 设置y轴的刻度标签
    plt.yticks(rotation=0)
    # 设置x轴的标签
    plt.xlabel(x_label, fontweight='bold', size=15)
    # 设置x轴的范围
    plt.xlim([2.5, 10.5])
    # 设置y轴的标签
    plt.ylabel(y_label, fontweight='bold', size=15)
    # 设置y轴的范围
    plt.ylim([0, 1.2e6])
    for _ in range(y_number):
        if _ == 0:
            plt.plot(x_data, y_data[_], label=label[_],
                     color=color[_], linestyle=linestyle[_], linewidth=2)
        else:
            plt.scatter(x_point, y_data[_], color='none', label=label[_],
                        edgecolors=color[_], marker=marker[_], s=20)
    # 设置网格线
    plt.grid(axis='both', color='gray', linestyle='--')
    plt.legend(loc=1, prop=font1)
    # plt.legend(bbox_to_anchor=(0.0, 1.0, 1.0, 0.0), loc='lower left',
    #            ncol=4, mode="expand", borderaxespad=0., prop=font1)
    fig = plt.gcf()
    # 设置图片的长和宽（英寸）
    fig.set_size_inches(7, 3.5)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print('已将图像保存为'+save_path)
    plt.show()


def get_xpoint_ypoint():
    '''
    从ERα_activity.xlsx文件读取数据
    '''
    # 打开xls文件
    xlsfile = pwd+"/ERα_activity.xlsx"
    book = xlrd.open_workbook(xlsfile)
    # 打开工作表
    sheet = book.sheet_by_index(0)
    sheet_name = book.sheet_names()[0]
    print("正在读取"+str(xlsfile)+"中的"+str(sheet_name)+"工作表")
    x_point = [0 for _ in range(smiles_nb)]
    y_point = [0 for _ in range(smiles_nb)]
    for i in range(smiles_nb):
        y_point[i] = sheet.cell_value(i+1, 1)
        x_point[i] = sheet.cell_value(i+1, 2)
    return x_point, y_point


def get_smooth_xy(x, y, density=50):
    """
    对散点进行平滑处理
    """
    smooth_x = np.linspace(min(x), max(x), density)
    smooth_y = scinpo.make_interp_spline(x, y)(smooth_x)
    return smooth_x, smooth_y


def plot_line5(x_data, y_data, save_path, x_label, y_label,
               var_orient='horizon'):
    """
    绘制loss变化曲线图
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
    label = ['Quantitative Prediction']
    # HTML颜色名 'red', 'green', 'blue', 'yellow', 'black', 'white',
    # 'cyan', 'darksalmon', 'gold', 'crimson'
    color = ['blue']
    # 线条风格 '-', '--', '-.', ':', 'None'
    linestyle = ['-']
    # 线条标记 'o', 'd', 'D', 'h', 'H', '_', '8', 'p', ',', '.', 's',
    #  '+', '*', 'x', '^', 'v', '<', '>', '|'
    marker = ['o']
    if y_number != len(label) or y_number != len(color) or \
            y_number != len(linestyle) or y_number != len(marker):
        print('y_data的label、color、linestyle、marker没有被正确设置！')
        exit(1)
    # 设置字体和大小
    mpl.rcParams.update({'font.family': 'Times New Roman',
                         'mathtext.fontset': 'stix'})
    plt.rc('font', family='Times New Roman', size=15)
    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}
    # plt.title('title')
    # 获取当前Axes对象
    ax = plt.gca()
    # 设置x轴的主刻度
    ax.xaxis.set_major_locator(MultipleLocator(10))
    # 设置x轴的副刻度
    # ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    # 设置x轴的刻度标签
    plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], rotation=0)
    # 设置y轴的主刻度
    ax.yaxis.set_major_locator(MultipleLocator(1e-2))
    # 设置y轴的副刻度
    # ax.yaxis.set_minor_locator(MultipleLocator(1))
    # 设置x轴的刻度标签
    plt.xticks(rotation=0)
    # 设置y轴的刻度标签
    plt.yticks(rotation=0)
    # 设置x轴的标签
    plt.xlabel(x_label, fontweight='bold', size=15)
    # 设置x轴的范围
    plt.xlim([0, 100])
    # 设置y轴的标签
    plt.ylabel(y_label, fontweight='bold', size=15)
    # 设置y轴的范围
    plt.ylim([0, 5e-2])
    for _ in range(y_number):
        sx, sy = get_smooth_xy(x_data, y_data[_])
        plt.plot(sx, sy, label=label[_], color=color[_],
                 linestyle=linestyle[_], linewidth=2,
                 marker=marker[_], markevery=[],
                 markerfacecolor=color[_], markeredgecolor='none')
        plt.scatter(x_data, y_data[_], color='none',
                    edgecolors=color[_], marker=marker[_], s=30)
    # 设置网格线
    plt.grid(axis='y', color='gray', linestyle='--')
    plt.legend(loc=1, prop=font1)
    fig = plt.gcf()
    # 设置图片的长和宽（英寸）
    fig.set_size_inches(7, 3.5)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print('已将图像保存为'+save_path)
    plt.show()


def plot_line6(x_data, y_data, save_path, x_label, y_label,
               var_orient='horizon'):
    """
    绘制loss变化曲线图
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
    label = ['Classification Prediction (Caco-2)',
             'Classification Prediction (CYP3A4)',
             'Classification Prediction (hERG)',
             'Classification Prediction (HOB)',
             'Classification Prediction (MN)']
    # HTML颜色名 'red', 'green', 'blue', 'yellow', 'black', 'white',
    # 'cyan', 'darksalmon', 'gold', 'crimson'
    color = ['red', 'green', 'blue', 'darksalmon', 'black']
    # 线条风格 '-', '--', '-.', ':', 'None'
    linestyle = [':', '-', '--', '-.', ':']
    # 线条标记 'o', 'd', 'D', 'h', 'H', '_', '8', 'p', ',', '.', 's',
    #  '+', '*', 'x', '^', 'v', '<', '>', '|'
    marker = ['o', 'd', 'D', 'h', 'H']
    if y_number != len(label) or y_number != len(color) or \
            y_number != len(linestyle) or y_number != len(marker):
        print('y_data的label、color、linestyle、marker没有被正确设置！')
        exit(1)
    # 设置字体和大小
    mpl.rcParams.update({'font.family': 'Times New Roman',
                         'mathtext.fontset': 'stix'})
    plt.rc('font', family='Times New Roman', size=15)
    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}
    # plt.title('title')
    # 获取当前Axes对象
    ax = plt.gca()
    # 设置x轴的主刻度
    ax.xaxis.set_major_locator(MultipleLocator(10))
    # 设置x轴的副刻度
    # ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    # 设置x轴的刻度标签
    plt.xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], rotation=0)
    # 设置y轴的主刻度
    ax.yaxis.set_major_locator(MultipleLocator(6e-2))
    # 设置y轴的副刻度
    # ax.yaxis.set_minor_locator(MultipleLocator(1))
    # 设置x轴的刻度标签
    plt.xticks(rotation=0)
    # 设置y轴的刻度标签
    plt.yticks(rotation=0)
    # 设置x轴的标签
    plt.xlabel(x_label, fontweight='bold', size=15)
    # 设置x轴的范围
    plt.xlim([0, 100])
    # 设置y轴的标签
    plt.ylabel(y_label, fontweight='bold', size=15)
    # 设置y轴的范围
    plt.ylim([0, 0.3])
    for _ in range(y_number):
        sx, sy = get_smooth_xy(x_data, y_data[_])
        plt.plot(sx, sy, label=label[_], color=color[_],
                 linestyle=linestyle[_], linewidth=2,
                 marker=marker[_], markevery=[],
                 markerfacecolor=color[_], markeredgecolor='none')
        # 'none'表示无填充颜色
        plt.scatter(x_data, y_data[_], color='none',
                    edgecolors=color[_], marker=marker[_], s=30)
    # 设置网格线
    plt.grid(axis='y', color='gray', linestyle='-')
    plt.legend(loc=1, prop=font1)
    fig = plt.gcf()
    # 设置图片的长和宽（英寸）
    fig.set_size_inches(7, 3.5)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        print('已将图像保存为'+save_path)
    plt.show()


if __name__ == '__main__':
    # plot_func1(-5, 5, 200, pwd+"/Figure/fun1.pdf", 'x', 'y')
    # plot_func2(-5, 5, 200, pwd+"/Figure/fun2.pdf", 'x', 'y')
    # plot_func3(-5, 5, 200, pwd+"/Figure/fun3.pdf", 'x', 'y')
    # x_point, y_point = get_xpoint_ypoint()
    # plot_fitting_curve(
    #     0, 10, 200, x_point, y_point, pwd+"/Figure/fun4.pdf",
    #     r'$pIC_{50}$', r'$IC_{50} (nM)$')
    x_data = [_+1 for _ in range(100)]
    y_data2 = dp.read_data(pwd+"/Question2_tr_loss.txt")
    y_data2 = dp.increase_ndim(y_data2, way='y')
    y_data3_1 = dp.read_data(pwd+"/Question3_1_tr_loss.txt")
    y_data3_2 = dp.read_data(pwd+"/Question3_2_tr_loss.txt")
    y_data3_3 = dp.read_data(pwd+"/Question3_3_tr_loss.txt")
    y_data3_4 = dp.read_data(pwd+"/Question3_4_tr_loss.txt")
    y_data3_5 = dp.read_data(pwd+"/Question3_5_tr_loss.txt")
    y_data = []
    y_data.append(y_data3_1)
    y_data.append(y_data3_2)
    y_data.append(y_data3_3)
    y_data.append(y_data3_4)
    y_data.append(y_data3_5)
    plot_line5(x_data, y_data2, pwd+"/Figure/fun5.pdf",
               'Epoch (#)', 'Train Loss')
    plot_line6(x_data, y_data, pwd+"/Figure/fun6.pdf",
               'Epoch (#)', 'Train Loss')
