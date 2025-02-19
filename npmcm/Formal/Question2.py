#!/usr/bin/python3
# -*- coding:utf-8 -*-
import os

import math
from numpy.lib.scimath import logn, power, sqrt
import numpy as np
import torch
import torch.nn as nn
import xlrd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.optim import Adam
from xlutils.copy import copy

import data_process as dp

BATCH_SIZE = 16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def swish(x):
    return x * torch.sigmoid(x)


class BP_network2(nn.Module):
    '''
    BP神经网络模型
    '''

    def __init__(self, sp=False):
        '''
        定义变量和常量
        '''
        # SMILES化合物的数量（训练）
        self.smiles_nb_tr = 1974
        # SMILES化合物的数量（测试）
        self.smiles_nb_te = 50
        # 分子描述符的数量
        self.descriptor_nb = 729
        # 使用IC50表示的生物活性的最值（训练）
        self.IC50_nM_tr_max = 0
        self.IC50_nM_tr_min = 0
        # 使用pIC50表示的生物活性的最值（测试）
        self.pIC50_tr_max = 0
        self.pIC50_tr_min = 0
        # 分子描述符的最值（训练）
        self.descriptor_tr_max = []
        self.descriptor_tr_min = []
        # 使用IC50表示的生物活性（测试）
        self.IC50_nM_te = [0.0 for _ in range(self.smiles_nb_te)]
        # 使用pIC50表示的生物活性（测试）
        self.pIC50_te = [0.0 for _ in range(self.smiles_nb_te)]
        # 初始化BP神经网络模型
        super().__init__()
        self.fc1 = nn.Linear(20, 16)
        self.b1 = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(16, 12)
        self.b2 = nn.BatchNorm1d(12)
        self.fc3 = nn.Linear(12, 8)
        self.b3 = nn.BatchNorm1d(8)
        self.fc4 = nn.Linear(8, 4)
        self.b4 = nn.BatchNorm1d(4)
        self.fc5 = nn.Linear(4, 1)
        # 确定当前目录
        self.pwd = os.getcwd()
        res = self.pwd.split('/')
        if sp is False:
            if res[-1] == "npmcm":
                self.pwd += "/Formal"
            elif res[-1] == "reuse-code":
                self.pwd += "/npmcm/Formal"

    def forward(self, x):
        x = swish(self.fc1(x))
        x = self.b1(x)
        x = swish(self.fc2(x))
        x = self.b2(x)
        x = swish(self.fc3(x))
        x = self.b3(x)
        x = swish(self.fc4(x))
        x = self.b4(x)
        x = torch.sigmoid(self.fc5(x))
        x = x.squeeze(-1)
        return x

    def write_xls(self):
        '''
        将数据写入xls文件
        '''
        # 打开xls文件
        xlsfile = self.pwd+"/ERα_activity.xlsx"
        book = xlrd.open_workbook(xlsfile)
        # 打开工作表
        new_book = copy(book)
        sheet = new_book.get_sheet(1)
        sheet_name = book.sheet_names()[1]
        print("将数据保存至"+str(xlsfile)+"中的"+str(sheet_name)+"工作表")
        for i in range(self.smiles_nb_te):
            sheet.write(i+1, 1, str(self.IC50_nM_te[i]))
            sheet.write(i+1, 2, str(self.pIC50_te[i]))
        new_book.save(xlsfile)


def train(model, x, y, optimizer, criterion):
    model.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    return loss, output


def solve(save_loss=False):
    net = BP_network2()
    net = net.double()
    net.to(device)
    EPOCHS = 100
    optm = Adam(net.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    filename = net.pwd+"/ERα_activity_train.csv"
    data = dp.read_data(filename)
    data = MinMaxScaler().fit_transform(np.array(data))
    data = np.array(data)
    data_x = data[:, 2:]
    data_y = data[:, 1]
    train_x, valid_x, train_y, valid_y = train_test_split(
        data_x, data_y, test_size=0.2, random_state=1)
    size_train = train_x.shape[0]
    train_x = torch.from_numpy(train_x)
    valid_x = torch.from_numpy(valid_x)
    train_y = torch.from_numpy(train_y)
    valid_y = torch.from_numpy(valid_y)
    loss_array = []
    for epoch in range(EPOCHS):
        net.train()
        epoch_loss = 0
        rem = size_train % BATCH_SIZE
        loop_nb = int((size_train-rem)/BATCH_SIZE)
        index = torch.randperm(size_train)
        for batch in range(loop_nb):
            i = index[batch*BATCH_SIZE:batch*BATCH_SIZE+BATCH_SIZE]
            x_train = train_x[i, :]
            y_train = train_y[i]
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            loss, predictions = train(net, x_train, y_train, optm, criterion)
            epoch_loss += loss.item()
        print('Epoch {} train Loss : {}'.format((epoch+1), epoch_loss/loop_nb))
        loss_array.append(epoch_loss/loop_nb)
        if save_loss is False:
            net.eval()
            epoch_val_loss = 0
            with torch.no_grad():
                x_test = valid_x
                y_test = valid_y
                x_test = x_test.to(device)
                y_test = y_test.to(device)
                output = net(x_test)
                epoch_val_loss = criterion(output, y_test)
            print('Epoch {} val Loss : {}'.format((epoch + 1), epoch_val_loss))
    if save_loss is False:
        save_path = net.pwd+"/Question2.pth"
        torch.save(net.state_dict(), save_path)
    if save_loss is True:
        filename = net.pwd+"/Question2_tr_loss.txt"
        dp.write_data(loss_array, filename)


def log_function(pIC50):
    '''
    IC50关于pIC50的负对数函数
    '''
    IC50 = 75.75224+1.86394e6/(sqrt(2*math.pi)*0.25985*pIC50) * \
        power(math.e, -0.5*((logn(math.e, (pIC50/2.38198))/0.25985)**2))
    return IC50


def test():
    net = BP_network2()
    net = net.double()
    filename = net.pwd+"/ERα_activity_train.csv"
    data = dp.read_data(filename)
    net.pIC50_tr_max = np.max(np.transpose(data)[1])
    net.pIC50_tr_min = np.min(np.transpose(data)[1])
    for _ in range(20):
        net.descriptor_tr_max.append(np.max(np.transpose(data)[_+2]))
        net.descriptor_tr_min.append(np.min(np.transpose(data)[_+2]))
    net.load_state_dict(torch.load(net.pwd+"/Question2.pth"))
    filename = net.pwd+"/ERα_activity_test.csv"
    data = dp.read_data(filename)
    data = dp.scaler_data(data, net.descriptor_tr_max,
                          net.descriptor_tr_min, numpy=True)
    output = net(torch.from_numpy(data))
    output = output.tolist()
    for _ in range(net.smiles_nb_te):
        net.pIC50_te[_] = output[_]
    net.pIC50_te = dp.recover_scaler_data(
        net.pIC50_te, [net.pIC50_tr_max], [net.pIC50_tr_min])
    net.pIC50_te = dp.reduce_ndim(net.pIC50_te)
    for _ in range(net.smiles_nb_te):
        net.IC50_nM_te[_] = log_function(net.pIC50_te[_])
    net.IC50_nM_te = dp.reduce_ndim(net.IC50_nM_te)
    net.write_xls()


if __name__ == '__main__':
    solve(True)
    # test()
