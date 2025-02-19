#!/usr/bin/python3
# -*- coding:utf-8 -*-
import csv
import os
from string import digits

import numpy as np
from numpy.core.fromnumeric import transpose


def conversion(data):
    """
    list类型和tuple类型之间的互相转换
    """
    if isinstance(data, list):
        data = tuple(data)
    elif isinstance(data, tuple):
        data = list(data)
    return data


def ndarray_to_list(data):
    """
    将ndarray类型转换为list类型
    """
    if isinstance(data, np.ndarray):
        data = data.tolist()
    return data


def list_to_ndarray(data):
    """
    将list类型转换为ndarray类型
    """
    if isinstance(data, list):
        data = np.array(data)
    return data


def get_ndim_shape(data):
    """
    判断矩阵（向量）的维度和形状
    """
    ndim = 0
    shape = []
    if isinstance(data, list):
        data = list_to_ndarray(data)
    if isinstance(data, np.ndarray):
        shape = conversion(data.shape)
        ndim = len(shape)
    return ndim, shape


def read_data(path, numpy=False):
    """
    从文本中读取数据
    """
    suffix = os.path.splitext(path)[1]
    f = open(path, 'r')
    data = []
    if suffix == '.csv':
        reader = csv.reader(f)
        for i in reader:
            temp = []
            for j in i:
                temp.append(float(j))
            data.append(temp)
        if numpy is True:
            data = list_to_ndarray(data)
    else:
        data = np.loadtxt(path)
        if numpy is False:
            data = ndarray_to_list(data)
    f.close()
    # print(data)
    return data


def write_data(data, path):
    """
    将数据写入文本
    """
    suffix = os.path.splitext(path)[1]
    f = open(path, 'w')
    if suffix == '.csv':
        if isinstance(data, list) or isinstance(data, tuple) or \
                isinstance(data, np.ndarray):
            writer = csv.writer(f)
            for _ in data:
                writer.writerow(_)
        else:
            print('csv文件用于保存list类型或tuple类型或ndarray类型！')
    else:
        data = list_to_ndarray(data)
        np.savetxt(path, data, fmt='%.4e', delimiter='\t')
    f.close()


def increase_ndim(data, way='x', numpy=False):
    """
    增加矩阵的维度
    """
    # way表示维度增加的方式
    # x表示 [a b]->[[a] [b]]
    # y表示 [a b]->[[a b]]
    if way == 'x':
        ndim, shape = get_ndim_shape(data)
        data = list_to_ndarray(data)
        data = np.expand_dims(data, axis=ndim)
    elif way == 'y':
        data = ndarray_to_list(data)
        res = []
        res.append(data)
        data = res
    else:
        print('way的值错误！')
        exit(1)
    if numpy is True:
        data = list_to_ndarray(data)
    elif numpy is False:
        data = ndarray_to_list(data)
    return data


def reduce_ndim(data, numpy=False):
    """
    减少矩阵的维度
    """
    data = list_to_ndarray(data)
    data = np.squeeze(data)
    if numpy is False:
        data = ndarray_to_list(data)
    return data


def combine_matrix(data1, data2, orient='y', numpy=False):
    """
    将两个矩阵合并
    """
    # orient表示合并方向
    # x表示 data1 + data2
    #      data1
    # y表示   +
    #      data2
    if orient != 'x' and orient != 'y':
        print('var_orient的值错误!')
        exit(1)
    ndim1, shape1 = get_ndim_shape(data1)
    ndim2, shape2 = get_ndim_shape(data2)
    if ndim1 == 1:
        data1 = increase_ndim(data1, orient, True)
        ndim1, shape1 = get_ndim_shape(data1)
    if ndim2 == 1:
        data2 = increase_ndim(data2, orient, True)
        ndim2, shape2 = get_ndim_shape(data2)
    if orient == 'x':
        if shape1[0] != shape2[0]:
            print('data1和data2的高应该相等!')
            exit(1)
        data = np.hstack((data1, data2))
    elif orient == 'y':
        if shape1[1] != shape2[1]:
            print('data1和data2的宽应该相等!')
            exit(1)
        data = np.vstack((data1, data2))
    if numpy is False:
        data = ndarray_to_list(data)
    return data


def partition_matrix(data, xmin, xmax, ymin, ymax, numpy=False):
    """
    从矩阵中分割出一个区域矩阵
    """
    # [xmin,xmax]指的是水平方向的范围（从0开始）xmax=-1表示x轴方向结束
    # [ymin,ymax]指的是垂直方向的范围（从0开始）ymax=-1表示y轴方向结束
    ndim, shape = get_ndim_shape(data)
    if ndim != 2:
        print('data应该是二维矩阵!')
        exit(1)
    if xmax != -1 and xmin > xmax:
        print('xmin<=xmax不满足!')
        exit(1)
    if ymax != -1 and ymin > ymax:
        print('ymin<=ymax不满足!')
        exit(1)
    if xmin < 0:
        xmin = 0
    if xmax == -1:
        xmax = shape[1]-1
    if xmax > shape[1]-1:
        xmax = shape[1]-1
    if ymin < 0:
        ymin = 0
    if ymax == -1:
        ymax = shape[0]-1
    if ymax > shape[0]-1:
        ymax = shape[0]-1
    data = ndarray_to_list(data)
    data = data[ymin:ymax+1]
    res = []
    for _ in range(ymax-ymin+1):
        res.append(data[_][xmin:xmax+1])
    res = reduce_ndim(res)
    if numpy is True:
        res = list_to_ndarray(res)
    return res


def extract_matrix_row(data, index, numpy=False):
    '''
    按照行序号index从矩阵data中提取矩阵res
    '''
    res = []
    data = ndarray_to_list(data)
    for _ in range(len(index)):
        res.append(data[index[_]])
    if numpy is True:
        res = list_to_ndarray(res)
    return res


def delete_matrix_row(data, index, numpy=False):
    '''
    按照行序号index从矩阵data中删除几行
    '''
    data = ndarray_to_list(data)
    save = [_ for _ in range(len(data))]
    for _ in index:
        save.remove(_)
    return extract_matrix_row(data, save, numpy)


def extract_matrix_column(data, index, numpy=False):
    '''
    按照列序号index从矩阵data中提取矩阵res
    '''
    res = []
    data = np.transpose(data)
    data = ndarray_to_list(data)
    for _ in range(len(index)):
        res.append(data[index[_]])
    res = np.transpose(res)
    if numpy is False:
        res = ndarray_to_list(res)
    return res


def delete_matrix_column(data, index, numpy=False):
    '''
    按照列序号index从矩阵data中删除几列
    '''
    data = ndarray_to_list(data)
    save = [_ for _ in range(len(data[0]))]
    for _ in index:
        save.remove(_)
    return extract_matrix_column(data, save, numpy)


def scaler_data(data, max, min, numpy=False):
    '''
    对数据进行标准化
    '''
    res = []
    data = np.transpose(data)
    ndim, shape = get_ndim_shape(data)
    if ndim == 1:
        data = increase_ndim(data, 'y', numpy=True)
    data = ndarray_to_list(data)
    for i in range(len(data)):
        for j in range(len(data[0])):
            if max[i]-min[i] != 0:
                data[i][j] = (data[i][j]-min[i])/(max[i]-min[i])
    res = np.transpose(data)
    if numpy is False:
        res = ndarray_to_list(res)
    return res


def recover_scaler_data(data, max, min, numpy=False):
    '''
    恢复标准化的数据
    '''
    res = []
    data = np.transpose(data)
    ndim, shape = get_ndim_shape(data)
    if ndim == 1:
        data = increase_ndim(data, 'y', numpy=True)
    data = ndarray_to_list(data)
    for i in range(len(data)):
        for j in range(len(data[0])):
            if max[i]-min[i] != 0:
                data[i][j] = data[i][j]*(max[i]-min[i])+min[i]
    res = np.transpose(data)
    if numpy is False:
        res = ndarray_to_list(res)
    return res
