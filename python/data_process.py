#!/usr/bin/python3
# -*- coding:utf-8 -*-
import copy
import csv
import json
import math
import os
import random
import sys
from decimal import Decimal
from string import digits

import numpy as np
from PyPDF2 import PdfReader, PdfWriter

data_dir = 'test_data/'


def check_work_dir():
    curr_dir = os.getcwd()
    file_path = sys.argv[0]
    # file_name = os.path.basename(file_path)
    # file_name = file_path.split('/')[-1]
    if file_path[0] == '/':
        work_dir = file_path
    else:
        if file_path[0] == '.' and file_path[1] == '/':
            work_dir = os.path.join(curr_dir, file_path[1:])
            # work_dir = curr_dir+file_path[1:]
        else:
            work_dir = os.path.join(curr_dir, file_path)
            # work_dir = curr_dir+'/'+file_path
    work_dir = os.path.dirname(work_dir)
    # work_dir = work_dir[:-(len(file_name))]
    if os.path.exists(work_dir):
        os.chdir(work_dir)


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
    elif suffix == '.json':
        data = json.load(f)
    else:
        data = np.loadtxt(path)
        ndim, shape = get_ndim_shape(data)
        if ndim == 1:
            data = [data]
        if numpy is False:
            data = ndarray_to_list(data)
    f.close()
    print(data)
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
    elif suffix == '.json':
        if isinstance(data, dict):
            json_str = json.dumps(data, indent=4)
            f.write(json_str)
        else:
            print('json文件用于保存dict类型！')
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


def extract_x_y(data, var_orient='horizon', numpy=False):
    """
    从数据矩阵中提取向量x和矩阵y
    """
    # var_orient指的是x轴数据的变化方向
    # x轴的数据默认出现在第一行（列）
    ndim, shape = get_ndim_shape(data)
    if ndim != 2:
        print('data应该是二维矩阵!')
        exit(1)
    if var_orient != 'vertical' and var_orient != 'horizon':
        print('var_orient的值错误!')
        exit(1)
    if var_orient == 'vertical':
        data = np.transpose(data)
    x_data = data[0:1][0]
    y_data = data[1:]
    if var_orient == 'vertical':
        y_data = np.transpose(y_data)
    if numpy is True:
        x_data = list_to_ndarray(x_data)
        y_data = list_to_ndarray(y_data)
    else:
        x_data = ndarray_to_list(x_data)
        y_data = ndarray_to_list(y_data)
    return x_data, y_data


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


def read_dir_data(dirname, var_orient='horizon', numpy=False):
    """
    将指定目录下的所有文件（数据矩阵）联合起来
    """
    files = os.listdir(dirname)
    y_data = []
    for _ in files:
        _ = dirname+_
        print('正在读取'+_+'...')
        data = read_data(_)
        x_data, _y_data = extract_x_y(data, var_orient)
        for __ in range(len(_y_data)):
            y_data.append(_y_data[__])
    if numpy is True:
        x_data = list_to_ndarray(x_data)
        y_data = list_to_ndarray(y_data)
    return x_data, y_data


def random_matrix(height, width, min, max, numpy=False):
    """
    随机生成指定高宽的二维矩阵
    """
    if height <= 0 or width <= 0:
        print('height和width的值错误!')
        exit(1)
    if min > max:
        print('min<=max不满足!')
        exit(1)
    data = []
    for _ in range(height):
        temp = []
        for __ in range(width):
            temp.append(random.uniform(min, max))
        data.append(temp)
    if numpy is True:
        data = list_to_ndarray(data)
    return data


def reverse_list(list):
    """
    列表倒置
    """
    return list[::-1]


def sort_matrix(data, line, row=True, ascend=True, numpy=False):
    """
    对矩阵按行（列）升（降）序排列
    """
    # line表示指定的行（列）号（从0开始）
    # row=True表示按行
    # row=False表示按列
    # ascend=True表示从左到右（从上到下）升序
    # ascend=False表示从左到右（从上到下）降序
    data = list_to_ndarray(data)
    if row is True:
        index = np.lexsort(data[0:line+1, :])
    else:
        index = np.lexsort(data.T[0:line+1, :])
    if ascend is False:
        index = reverse_list(ndarray_to_list(index))
    if row is True:
        data = data.T[index].T
    else:
        data = data[index, :]
    if numpy is False:
        data = ndarray_to_list(data)
    return data


def singular_data(data):
    """
    识别奇异数据
    """
    max_gap = 0
    singular = 0
    temp = copy.deepcopy(data)
    for _ in data:
        temp.remove(_)
        gap = math.fabs(_-np.mean(temp))
        if gap > max_gap:
            max_gap = gap
            singular = _
        temp.append(_)
    data.remove(singular)
    return data


def remove_singular_data(data, nb=1, numpy=False):
    """
    去除列表中的奇异数据
    """
    # nb表示奇异数据的个数
    data = ndarray_to_list(data)
    if nb >= len(data):
        print('nb的值不能超过列表的长度!')
        exit(1)
    for _ in range(nb):
        data = singular_data(data)
    if numpy is True:
        data = list_to_ndarray(data)
    return data


def cal_average_data(dirname, numpy=False):
    """
    计算指定目录下的所有文件（数据矩阵）的平均值
    """
    files = os.listdir(dirname)
    temp = read_data(dirname+files[0])
    height = len(temp)
    width = len(temp[0])
    data = [[0 for j in range(width)] for i in range(height)]
    for i in range(height):
        for j in range(width):
            file_data = []
            for _ in files:
                _ = dirname+_
                temp = read_data(_)
                file_data.append(temp[i][j])
            file_data = remove_singular_data(file_data)
            data[i][j] = np.mean(file_data)
    filename = files[0]
    remove_digits = str.maketrans('', '', digits)
    filename = filename.translate(remove_digits)
    write_data(data, dirname+filename)
    if numpy is False:
        data = ndarray_to_list(data)
    return data


def crop_pdf(input_path, output_path, lower_ratio, left_ratio, upper_ratio, right_ratio):
    #! pip3 install PyPDF2
    with open(input_path, 'rb') as input_file:
        pdf = PdfReader(input_file)
        first_page = pdf.pages[0]
        width = first_page.mediabox.width
        height = first_page.mediabox.height

        target_width = width*Decimal(1-left_ratio-right_ratio)
        target_height = height*Decimal(1-lower_ratio-upper_ratio)
        x_offset = width*Decimal(left_ratio)
        y_offset = height*Decimal(lower_ratio)

        output = PdfWriter()
        for page_num in range(len(pdf.pages)):
            page = pdf.pages[page_num]
            page.cropbox.lower_left = (x_offset, y_offset)
            page.cropbox.upper_right = (
                x_offset + target_width, y_offset + target_height)
            output.add_page(page)

        with open(output_path, 'wb') as output_file:
            output.write(output_file)


def fluctuate_number(original_number, ratio=0.01):
    lower_bound = original_number * (1-ratio)
    upper_bound = original_number * (1+ratio)
    new_number = random.uniform(lower_bound, upper_bound)
    return new_number


def example():
    json = read_data(data_dir+'read.json')
    # print(json['float'])
    # print(json['string'])
    # print(json['bool'])
    # print(json['multi_array']['string2'])
    # print(json['multi_object'][1]['name'])
    csv = read_data(data_dir+'read.csv', True)
    txt = read_data(data_dir+'read.txt', True)
    # txt = 3*txt-10
    # csv = 100/csv
    print(np.transpose(txt))
    print(get_ndim_shape(txt))
    x = np.transpose([[1, 2, 3, 4], [11, 12, 13, 14]])
    data = combine_matrix(x, txt, orient='x', numpy=True)
    print(data)
    x_data, y_data = extract_x_y(data, numpy=True)
    print(x_data)
    print(y_data)
    part = partition_matrix(data, 1, 4, 1, 3, True)
    print(part)
    write_data(json, data_dir+'write.json')
    write_data(csv, data_dir+'write.csv')
    write_data(txt, data_dir+'write.txt')
    x_data, y_data = read_dir_data(data_dir+'joint_data/', numpy=True)
    print(x_data)
    print(y_data)
    t = [[1, 2, 8, 6, 7], [3, 1, 4, 7, 2], [8, 3, 9, 4, 12]]
    print(sort_matrix(t, 2, True, False, True))
    print(cal_average_data(data_dir+'average_data/', True))


if __name__ == '__main__':
    check_work_dir()
    example()
