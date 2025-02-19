#!/usr/bin/python3
# -*- coding:utf-8 -*-
import os

import numpy as np
import torch
import torch.nn as nn
import xlrd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.optim import Adam, SGD
from xlutils.copy import copy

import data_process as dp

BATCH_SIZE = 16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def swish(x):
    return x * torch.sigmoid(x)


class BP_network3(nn.Module):
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
        # ADMET性质的数量
        self.ADMET_nb = 5
        # ADMET性质的最值（训练）
        self.ADMET_tr_max = 0
        self.ADMET_tr_min = 0
        # 分子描述符的最值（训练）
        self.descriptor_tr_max = []
        self.descriptor_tr_min = []
        # ADMET性质的取值（测试）
        self.ADMET_te = [0 for _ in range(self.smiles_nb_te)]
        # 初始化BP神经网络模型
        super().__init__()
        self.fc1 = nn.Linear(729, 1024)
        self.b1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.b2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.b3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 128)
        self.b4 = nn.BatchNorm1d(128)
        self.fc5 = nn.Linear(128, 64)
        self.b5 = nn.BatchNorm1d(64)
        self.fc6 = nn.Linear(64, 32)
        self.b6 = nn.BatchNorm1d(32)
        self.fc7 = nn.Linear(32, 16)
        self.b7 = nn.BatchNorm1d(16)
        self.fc8 = nn.Linear(16, 8)
        self.b8 = nn.BatchNorm1d(8)
        self.fc9 = nn.Linear(8, 4)
        self.b9 = nn.BatchNorm1d(4)
        self.fc10 = nn.Linear(4, 1)
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
        x = swish(self.fc5(x))
        x = self.b5(x)
        x = swish(self.fc6(x))
        x = self.b6(x)
        x = swish(self.fc7(x))
        x = self.b7(x)
        x = swish(self.fc8(x))
        x = self.b8(x)
        x = swish(self.fc9(x))
        x = self.b9(x)
        x = torch.sigmoid(self.fc10(x))
        return x

    def write_xls(self, model_idx):
        '''
        将数据写入xls文件
        '''
        # 打开xls文件
        xlsfile = self.pwd+"/ADMET.xlsx"
        book = xlrd.open_workbook(xlsfile)
        # 打开工作表
        new_book = copy(book)
        sheet = new_book.get_sheet(1)
        sheet_name = book.sheet_names()[1]
        print("将数据保存至"+str(xlsfile)+"中的"+str(sheet_name)+"工作表")
        for _ in range(self.smiles_nb_te):
            sheet.write(_+1, model_idx, str(self.ADMET_te[_]))
        new_book.save(xlsfile)


def train(model, x, y, optimizer, criterion):
    model.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    return loss, output


def solve(model_idx, save_loss=False):
    net = BP_network3()
    net = net.double()
    net.to(device)
    EPOCHS = 100
    optm = SGD(net.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    filename = net.pwd+"/ADMET_train"+str(model_idx)+".csv"
    data = dp.read_data(filename)
    data = MinMaxScaler().fit_transform(np.array(data))
    data = np.array(data)
    data_x = data[:, 1:]
    data_y = data[:, :1]
    train_x, valid_x, train_y, valid_y = train_test_split(
        data_x, data_y, test_size=0.2, random_state=1)
    size_train = train_x.shape[0]
    size_valid = valid_x.shape[0]
    train_x = torch.from_numpy(np.array(train_x))
    valid_x = torch.from_numpy(np.array(valid_x))
    train_y = torch.from_numpy(np.array(train_y))
    valid_y = torch.from_numpy(np.array(valid_y))
    save_path = net.pwd+"/Question3_"+str(model_idx)+".pth"
    highest_acc = 0
    loss_array = []
    for epoch in range(EPOCHS):
        net.train()
        epoch_loss = 0
        acc = 0
        rem = size_train % BATCH_SIZE
        loop_nb = int((size_train-rem)/BATCH_SIZE)
        index = torch.randperm(size_train)
        for batch in range(loop_nb):
            i = index[batch*BATCH_SIZE:batch*BATCH_SIZE+BATCH_SIZE]
            x_train = train_x[i, :]
            y_train = train_y[i, :]
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            loss, predictions = train(net, x_train, y_train, optm, criterion)
            epoch_loss += loss.item()
        if save_loss is True:
            loss_array.append(epoch_loss/loop_nb)
            print('Model {} Epoch {} train Loss : {}'.format(
                model_idx, (epoch+1), epoch_loss/loop_nb))
        if save_loss is False:
            net.eval()
            with torch.no_grad():
                x_test = valid_x
                y_test = valid_y
                x_test = x_test.to(device)
                y_test = y_test.to(device)
                output = net(x_test)
                for _ in range(size_valid):
                    res = output[_][0]
                    if res > 0.5:
                        res = 1
                    else:
                        res = 0
                    if res == y_test[_]:
                        acc += 1
            acc = acc/size_valid
            # print('Model {} Epoch {} val Acc : {}'.format(
            #     model_idx, (epoch + 1), acc))
            if acc > highest_acc:
                highest_acc = acc
                torch.save(net.state_dict(), save_path)
    if save_loss is False:
        print('Model {} highest val Acc : {}'.format(
            model_idx, highest_acc))
    if save_loss is True:
        filename = net.pwd+"/Question3_"+str(model_idx)+"_tr_loss.txt"
        dp.write_data(loss_array, filename)


def test(model_idx):
    net = BP_network3()
    net = net.double()
    filename = net.pwd+"/ADMET_train"+str(model_idx)+".csv"
    data = dp.read_data(filename)
    for _ in range(net.descriptor_nb):
        net.descriptor_tr_max.append(np.max(np.transpose(data)[_+1]))
        net.descriptor_tr_min.append(np.min(np.transpose(data)[_+1]))
    net.load_state_dict(torch.load(
        net.pwd+"/Question3_"+str(model_idx)+".pth"))
    filename = net.pwd+"/ADMET_test"+str(model_idx)+".csv"
    data = dp.read_data(filename)
    data = dp.scaler_data(data, net.descriptor_tr_max,
                          net.descriptor_tr_min, numpy=True)
    output = net(torch.from_numpy(data))
    output = output.tolist()
    for _ in range(net.smiles_nb_te):
        net.ADMET_te[_] = output[_][0]
        if net.ADMET_te[_] > 0.5:
            net.ADMET_te[_] = 1
        else:
            net.ADMET_te[_] = 0
    net.write_xls(model_idx)


if __name__ == '__main__':
    solve(5, True)
    # for _ in range(5):
    #     test(_+1)
