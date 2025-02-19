#!/usr/bin/python3
# -*- coding:utf-8 -*-
import os
from random import randint, random, sample

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xlrd
from sklearn.preprocessing import MinMaxScaler

import data_process as dp
from Question1 import SMILES
from Question2 import BP_network2
from Question3 import BP_network3

SMILES_NB = 1974
DESCRIPTOR_NB = 729
BATCH_SIZE = 16
MAX_SAMPLE_NB = 200
# 淘汰率
ELIMI_RATIO = 0.3
# 均匀交叉比例
CROSS_RATIO = 0.6

"""
五种ADMET性质
Caco-2	CYP3A4	hERG	HOB	MN
较好的情况
1       1       0       1   0
"""


def loss_function(recon_x, x, mu, logvar):
    """
    损失函数
    """
    BCE_loss = nn.BCELoss(reduction='sum')
    reconstruction_loss = BCE_loss(recon_x, x)
    KL_divergence = -0.5 * torch.sum(1+logvar-torch.exp(logvar)-mu**2)
    # print(reconstruction_loss, KL_divergence)
    return reconstruction_loss + KL_divergence


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(DESCRIPTOR_NB, 400)
        self.fc2_mean = nn.Linear(400, DESCRIPTOR_NB)
        self.fc2_logvar = nn.Linear(400, DESCRIPTOR_NB)
        self.fc3 = nn.Linear(DESCRIPTOR_NB, 400)
        self.fc4 = nn.Linear(400, DESCRIPTOR_NB)
        self.pwd = os.getcwd()
        res = self.pwd.split('/')
        if res[-1] == "npmcm":
            self.pwd += "/Formal"
        elif res[-1] == "reuse-code":
            self.pwd += "/npmcm/Formal"

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc2_mean(h1), self.fc2_logvar(h1)

    def reparametrization(self, mu, logvar):
        std = 0.5 * torch.exp(logvar)
        z = torch.randn(std.size()) * std + mu
        return z

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrization(mu, logvar)
        return self.decode(z), mu, logvar


def train(vae, round_idx, iteration_idx, epoch_idx, optimizer, data_x):
    vae.train()
    vae.to("cpu")
    rem = SMILES_NB % BATCH_SIZE
    loop_nb = int((SMILES_NB - rem) / BATCH_SIZE)
    index = torch.randperm(SMILES_NB)
    for batch in range(loop_nb):
        i = index[batch * BATCH_SIZE:batch * BATCH_SIZE + BATCH_SIZE]
        x_train = data_x[i, :]
        x_train = x_train.to("cpu")
        gen_imgs, mu, logvar = vae(x_train)
        loss = loss_function(gen_imgs, x_train, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('round_idx {} iteration_idx {} Epoch {}'.format(
        round_idx, iteration_idx, epoch_idx+1))


def read_xls(pwd):
    '''
    从Molecular_Descriptor.xlsx文件读取数据
    '''
    # 打开xls文件
    xlsfile = pwd+"/Molecular_Descriptor.xlsx"
    book = xlrd.open_workbook(xlsfile)
    # 打开工作表
    sheet = book.sheet_by_index(0)
    sheet_name = book.sheet_names()[0]
    print("正在读取"+str(xlsfile)+"中的"+str(sheet_name)+"工作表")
    descriptor_tr = [[0 for _ in range(DESCRIPTOR_NB)]
                     for _ in range(SMILES_NB)]
    for i in range(SMILES_NB):
        for j in range(DESCRIPTOR_NB):
            descriptor_tr[i][j] = sheet.cell_value(i+1, j+1)
    return descriptor_tr


def select_sample(vae, data, generate_nb):
    """
    挑选generate_nb个样本以保证种群数量为MAX_SAMPLE_NB
    """
    index = sample(range(SMILES_NB), generate_nb)
    data = data.tolist()
    input = dp.extract_matrix_row(data, index, True)
    input = torch.from_numpy(input)
    gen_imgs, mu, logvar = vae(input)
    return gen_imgs


def judge_ADMET(model3_1, model3_2, model3_3, model3_4, model3_5, data):
    """
    判断所有样本的ADMET性质
    """
    elimi_index = []
    nb = len(data)
    good_nb = [0 for _ in range(nb)]
    data = torch.from_numpy(np.array(data))
    out = model3_1(data)
    for _ in range(nb):
        if out[_][0] > 0.5:
            good_nb[_] += 1
    out = model3_2(data)
    for _ in range(nb):
        if out[_][0] > 0.5:
            good_nb[_] += 1
    out = model3_3(data)
    for _ in range(nb):
        if out[_][0] < 0.5:
            good_nb[_] += 1
    out = model3_4(data)
    for _ in range(nb):
        if out[_][0] > 0.5:
            good_nb[_] += 1
    out = model3_5(data)
    for _ in range(nb):
        if out[_][0] < 0.5:
            good_nb[_] += 1
    for _ in range(nb):
        if good_nb[_] < 3:
            elimi_index.append(_)
    return elimi_index


def pro_fit(model2, smiles, species, curr_sample_nb):
    """
    计算当前种群每个样本的挑选概率
    """
    temp = dp.extract_matrix_column(species, smiles.feature_index, True)
    temp = torch.from_numpy(temp)
    out_temp = model2(temp)
    out_socres = []
    for _ in range(curr_sample_nb):
        out_socres.append(int(out_temp[_]*100))
    for _ in range(curr_sample_nb):
        if _ >= 1:
            out_socres[_] += out_socres[_-1]
    return out_socres


def get_index_by_random(best, select_pro, curr_sample_nb):
    """
    根据随机数得到的序号
    """
    for _ in range(curr_sample_nb):
        if _ >= 1:
            if best <= select_pro[_] and best > select_pro[_-1]:
                return _
        else:
            if best <= select_pro[_]:
                return _


def init_model():
    """
    初始化模型SMILES、VAE、BP_network2、BP_network3
    """
    smiles = SMILES()
    smiles.feature_selection1()
    vae = VAE()
    vae.double()
    model2 = BP_network2()
    model2.double()
    model2.load_state_dict(torch.load(vae.pwd+"/Question2.pth"))
    model3_1 = BP_network3()
    model3_1.double()
    model3_1.load_state_dict(torch.load(vae.pwd+"/Question3_1.pth"))
    model3_2 = BP_network3()
    model3_2.double()
    model3_2.load_state_dict(torch.load(vae.pwd+"/Question3_2.pth"))
    model3_3 = BP_network3()
    model3_3.double()
    model3_3.load_state_dict(torch.load(vae.pwd+"/Question3_3.pth"))
    model3_4 = BP_network3()
    model3_4.double()
    model3_4.load_state_dict(torch.load(vae.pwd+"/Question3_4.pth"))
    model3_5 = BP_network3()
    model3_5.double()
    model3_5.load_state_dict(torch.load(vae.pwd+"/Question3_5.pth"))
    return smiles, vae, model2, model3_1, \
        model3_2, model3_3, model3_4, model3_5


def genetic_algorithm(data, smiles, vae, model2, model3_1, model3_2,
                      model3_3, model3_4, model3_5,
                      round_idx, iteration, epoch=20):
    """
    遗传算法
    """
    # 当前种群中的样本数量
    curr_sample_nb = 0
    # 种群里的全部样本
    species = []
    if os.path.isfile(vae.pwd+"/selected_species.csv"):
        print("发现已存在的selected_species.csv")
        species = dp.read_data(vae.pwd+"/selected_species.csv")
        curr_sample_nb = len(species)
    print("初始curr_sample_nb="+str(curr_sample_nb))
    optimizer = torch.optim.Adam(vae.parameters(), lr=0.0003)
    data = MinMaxScaler().fit_transform(np.array(data))
    data = torch.from_numpy(np.array(data))
    for iteration_idx in range(iteration):
        if os.path.isfile(vae.pwd+"/Question4.pth"):
            print("发现已存在的Question4.pth")
            vae.load_state_dict(torch.load(vae.pwd+"/Question4.pth"))
        for epoch_idx in range(epoch):
            train(vae, round_idx, iteration_idx, epoch_idx, optimizer, data)
        torch.save(vae.state_dict(), vae.pwd+"/Question4.pth")
        generate_nb = MAX_SAMPLE_NB-curr_sample_nb
        output = select_sample(vae, data, generate_nb)
        if len(species) == 0:
            species = output.tolist()
        else:
            species = dp.combine_matrix(species, output.tolist())
        curr_sample_nb += generate_nb
        print("round_idx="+str(round_idx)+"iteration_idx="+str(iteration_idx) +
              " select curr_sample_nb="+str(curr_sample_nb))
        # 通过ADMET性质淘汰样本
        elimi_index = judge_ADMET(
            model3_1, model3_2, model3_3, model3_4, model3_5, species)
        curr_sample_nb -= len(elimi_index)
        species = dp.delete_matrix_row(species, elimi_index)
        print("round_idx="+str(round_idx)+"iteration_idx="+str(iteration_idx) +
              " ADMET curr_sample_nb="+str(curr_sample_nb))
        # 通过pIC50淘汰样本
        elimi_index = []
        temp = dp.extract_matrix_column(species, smiles.feature_index, True)
        temp = torch.from_numpy(temp)
        out_temp = model2(temp)
        out_temp_socres = []
        for _ in range(curr_sample_nb):
            out_temp_socres.append((_, out_temp[_]))
        out_temp_socres.sort(key=lambda x: x[1])
        elimi_nb = int(curr_sample_nb*ELIMI_RATIO)
        for _ in range(elimi_nb):
            elimi_index.append(out_temp_socres[_][0])
        curr_sample_nb -= len(elimi_index)
        species = dp.delete_matrix_row(species, elimi_index)
        print("round_idx="+str(round_idx)+"iteration_idx="+str(iteration_idx) +
              " pIC50 curr_sample_nb="+str(curr_sample_nb))
        # 挑选两位杰出者进行均匀交叉
        if curr_sample_nb > 1:
            while curr_sample_nb <= 150:
                select_pro = pro_fit(model2, smiles, species, curr_sample_nb)
                best_index = []
                best = randint(1, select_pro[-1])
                best_index.append(get_index_by_random(
                    best, select_pro, curr_sample_nb))
                while True:
                    best = randint(1, select_pro[-1])
                    temp = get_index_by_random(
                        best, select_pro, curr_sample_nb)
                    if best_index[0] != temp:
                        best_index.append(temp)
                        break
                print("iteration_idx="+str(iteration_idx) +
                      " best="+str(best_index))
                new_sample = [[0 for _ in range(DESCRIPTOR_NB)]
                              for _ in range(2)]
                for _ in range(DESCRIPTOR_NB):
                    pro = random()
                    if pro <= CROSS_RATIO:
                        new_sample[0][_] = species[best_index[1]][_]
                        new_sample[1][_] = species[best_index[0]][_]
                    else:
                        new_sample[0][_] = species[best_index[0]][_]
                        new_sample[1][_] = species[best_index[1]][_]
                curr_sample_nb += 2
                species = dp.combine_matrix(species, new_sample)
                print("iteration_idx="+str(iteration_idx) +
                      " cross curr_sample_nb="+str(curr_sample_nb))
    print("---------------------------已完成样本挑选-----------------------------")
    # 通过ADMET性质淘汰样本
    elimi_index = []
    elimi_index = judge_ADMET(
        model3_1, model3_2, model3_3, model3_4, model3_5, species)
    curr_sample_nb -= len(elimi_index)
    species = dp.delete_matrix_row(species, elimi_index)
    print("round_idx="+str(round_idx) + "curr_sample_nb=" +
          str(curr_sample_nb)+" " + str(len(species)))
    species_filename = vae.pwd+"/selected_species.csv"
    dp.write_data(species, species_filename)


def get_features_values(vae, smiles, model2, data, sample_nb, feature_nb):
    """
    得出最终结果
    """
    filename = vae.pwd+"/selected_species.csv"
    species = dp.read_data(filename)
    row_nb = len(species)
    column_nb = len(species[0])

    row_index = []
    temp = dp.extract_matrix_column(species, smiles.feature_index, True)
    temp = torch.from_numpy(temp)
    out_temp = model2(temp)
    out_socres = []
    for _ in range(row_nb):
        out_socres.append((_, out_temp[_]))
    out_socres.sort(key=lambda x: x[1], reverse=True)
    for _ in range(sample_nb):
        row_index.append(out_socres[_][0])
    row_index.sort()

    column_index = []
    relative_variance = []
    for _ in range(column_nb):
        species_column = dp.reduce_ndim(dp.extract_matrix_column(species, [_]))
        species_var = np.var(species_column)
        data_column = dp.reduce_ndim(dp.extract_matrix_column(data, [_]))
        data_var = np.var(data_column)
        if data_var != 0:
            relative_variance.append((_, species_var/data_var))
        else:
            relative_variance.append((_, 9999))
    relative_variance.sort(key=lambda x: x[1])
    for _ in range(feature_nb):
        column_index.append(relative_variance[_][0])
    column_index.sort()

    species = dp.extract_matrix_row(species, row_index)
    species = dp.extract_matrix_column(species, column_index)

    max_scaler = []
    min_scaler = []
    for _ in range(feature_nb):
        max_scaler.append(np.max((np.transpose(data))[column_index[_]]))
        min_scaler.append(np.min(((np.transpose(data))[column_index[_]])))
    species = dp.recover_scaler_data(species, max_scaler, min_scaler)
    filename = vae.pwd+"/Question4_results.csv"
    dp.write_data(species, filename)

    print("挑选"+str(feature_nb)+"个特征（所有的特征序号都是从0开始）")
    for _ in range(feature_nb):
        min = np.min(dp.reduce_ndim(dp.extract_matrix_column(species, [_])))
        max = np.max(dp.reduce_ndim(dp.extract_matrix_column(species, [_])))
        print("特征序号"+str(column_index[_])+" 特征名称" +
              str(smiles.descriptor_name[column_index[_]]) +
              " 取值范围"+str(min)+"~"+str(max))


if __name__ == '__main__':
    smiles, vae, model2, model3_1, model3_2,\
        model3_3, model3_4, model3_5 = init_model()
    data = read_xls(vae.pwd)
    # for round_idx in range(10):
    #     genetic_algorithm(data, smiles, vae, model2, model3_1, model3_2,
    #                       model3_3, model3_4, model3_5,
    #                       round_idx, iteration=5)
    get_features_values(vae, smiles, model2, data, 20, 20)
