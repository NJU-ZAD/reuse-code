#!/usr/bin/python3
# -*- coding:utf-8 -*-
from math import pi
import os

import numpy as np
import sklearn.feature_selection as fs
import xlrd
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import (RFE, SelectFromModel, SelectKBest,
                                       VarianceThreshold)
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import normalized_mutual_info_score

import data_process as dp


class SMILES:
    '''
    SMILES模型
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
        # 使用IC50表示的生物活性（训练）
        self.IC50_nM_tr = [0 for _ in range(self.smiles_nb_tr)]
        # 使用IC50表示的生物活性（测试）
        self.IC50_nM_te = [0 for _ in range(self.smiles_nb_te)]
        # 使用pIC50表示的生物活性（训练）
        self.pIC50_tr = [0 for _ in range(self.smiles_nb_tr)]
        # 使用pIC50表示的生物活性（测试）
        self.pIC50_te = [0 for _ in range(self.smiles_nb_te)]
        # 分子描述符的名称
        self.descriptor_name = []
        # 分子描述符的取值（训练）
        self.descriptor_tr = [
            [0 for _ in range(self.descriptor_nb)]
            for _ in range(self.smiles_nb_tr)]
        self.descriptor_tr_scaler = []
        # 分子描述符的取值（测试）
        self.descriptor_te = [
            [0 for _ in range(self.descriptor_nb)]
            for _ in range(self.smiles_nb_te)]
        # ADMET性质的名称
        self.ADMET_name = []
        # ADMET性质的取值（训练）
        self.ADMET_tr = [[0 for _ in range(self.ADMET_nb)]
                         for _ in range(self.smiles_nb_tr)]
        self.ADMET_tr_scaler = []
        # ADMET性质的取值（测试）
        self.ADMET_te = [[0 for _ in range(self.ADMET_nb)]
                         for _ in range(self.smiles_nb_te)]
        # 确定当前目录
        self.pwd = os.getcwd()
        res = self.pwd.split('/')
        if sp is False:
            if res[-1] == "npmcm":
                self.pwd += "/Formal"
            elif res[-1] == "reuse-code":
                self.pwd += "/npmcm/Formal"
        self.init_name()
        self.read_xls("ERα_activity.xlsx")
        self.read_xls("Molecular_Descriptor.xlsx")
        self.read_xls("Molecular_Descriptor.xlsx", type=1)
        self.read_xls("ADMET.xlsx")

    def init_name(self):
        '''
        初始化descriptor_name和ADMET_name
        '''
        xlsfile = self.pwd+"/Molecular_Descriptor.xlsx"
        book = xlrd.open_workbook(xlsfile)
        sheet = book.sheet_by_index(0)
        for i in range(self.descriptor_nb):
            self.descriptor_name.append(sheet.cell_value(0, i+1))
        xlsfile = self.pwd+"/ADMET.xlsx"
        book = xlrd.open_workbook(xlsfile)
        sheet = book.sheet_by_index(0)
        for i in range(self.ADMET_nb):
            self.ADMET_name.append(sheet.cell_value(0, i+1))
        # print(self.descriptor_name)
        # print(len(self.descriptor_name))
        # print(self.ADMET_name)
        # print(len(self.ADMET_name))

    def read_xls(self, filename, type=0):
        '''
        从xls文件读取数据
        filename表示要读取的文件名
        type=0 -> training工作表
        type=1 -> test工作表
        '''
        # 打开xls文件
        xlsfile = self.pwd+"/"+filename
        book = xlrd.open_workbook(xlsfile)
        # 打开工作表
        sheet = book.sheet_by_index(type)
        sheet_name = book.sheet_names()[type]
        print("正在读取"+str(xlsfile)+"中的"+str(sheet_name)+"工作表")
        if type == 0:
            smiles_nb = self.smiles_nb_tr
        elif type == 1:
            smiles_nb = self.smiles_nb_te

        if filename == "Molecular_Descriptor.xlsx":
            for i in range(smiles_nb):
                for j in range(self.descriptor_nb):
                    if type == 0:
                        self.descriptor_tr[i][j] = sheet.cell_value(i+1, j+1)
                    elif type == 1:
                        self.descriptor_te[i][j] = sheet.cell_value(i+1, j+1)
        elif filename == "ERα_activity.xlsx":
            for i in range(smiles_nb):
                if type == 0:
                    self.IC50_nM_tr[i] = sheet.cell_value(i+1, 1)
                    self.pIC50_tr[i] = sheet.cell_value(i+1, 2)
                elif type == 1:
                    self.IC50_nM_te[i] = sheet.cell_value(i+1, 1)
                    self.pIC50_te[i] = sheet.cell_value(i+1, 2)
        elif filename == "ADMET.xlsx":
            for i in range(smiles_nb):
                for j in range(self.ADMET_nb):
                    if type == 0:
                        self.ADMET_tr[i][j] = sheet.cell_value(i+1, j+1)
                    elif type == 1:
                        self.ADMET_te[i][j] = sheet.cell_value(i+1, j+1)

    def output_descriptor_name(self):
        print("20个特征对应的分子描述符为")
        for _ in range(20):
            print(self.descriptor_name[self.feature_index[_]])

    def feature_selection1(self):
        '''
        选择20个重要性最高的分子描述符
        '''
        self.descriptor_tr_scaler = MinMaxScaler().fit_transform(
            np.array(self.descriptor_tr))
        # 选择方差大于阈值的特征
        selector = VarianceThreshold(threshold=0.001)
        selector.fit(self.descriptor_tr_scaler)
        descriptor_tr = selector.transform(self.descriptor_tr_scaler)
        # print("各个特征的方差为\n"+str(selector.variances_))
        self.feature_index = selector.get_support(True)
        self.feature_nb = len(self.feature_index)
        print("方差选择法的特征序号为\n"+str(self.feature_index))
        print("方差选择法的特征数量为"+str(self.feature_nb))

        kmeans = KMeans(n_clusters=30, init='k-means++')
        kmeans.fit(np.transpose(descriptor_tr))
        # print(kmeans.labels_)
        feature_index_2d = [[] for _ in range(30)]
        for _ in range(self.feature_nb):
            type = kmeans.labels_[_]
            feature_index_2d[type].append(self.feature_index[_])
        print("kmeans的特征序号为（30类）\n"+str(feature_index_2d))
        for i in range(30):
            size = len(feature_index_2d[i])
            print("第"+str(i+1)+"类")
            for j in range(size):
                print(
                    str(feature_index_2d[i][j])+"-" +
                    str(self.descriptor_name[feature_index_2d[i][j]]))

        self.feature_index = []
        best = SelectKBest(score_func=fs.f_regression, k=1)
        for i in range(30):
            best_descriptor = dp.extract_matrix_column(
                self.descriptor_tr_scaler, feature_index_2d[i])
            best.fit(best_descriptor, self.pIC50_tr)
            temp_index = feature_index_2d[i][(best.get_support(True))[0]]
            self.feature_index.append(temp_index)
        print("相关系数法的特征序号为\n"+str(self.feature_index))
        self.feature_nb = len(self.feature_index)
        print("相关系数法的特征数量为"+str(self.feature_nb))

        socres = [0 for _ in range(self.feature_nb)]
        descriptor_tr = dp.extract_matrix_column(
            self.descriptor_tr_scaler, self.feature_index)
        signals = np.transpose(descriptor_tr)
        index_socres = []
        for _ in range(self.feature_nb):
            socres[_] = normalized_mutual_info_score(signals[_], self.pIC50_tr)
            index_socres.append((self.feature_index[_], socres[_]))
        index_socres.sort(key=lambda x: x[1], reverse=True)
        print("根据互信息法将"+str(self.feature_nb)+"个特征从高到低排序\n"+str(index_socres))
        feature_index = [0 for _ in range(20)]
        for _ in range(20):
            feature_index[_] = index_socres[_][0]
        feature_index.sort()
        self.feature_index = feature_index
        print("互信息法的特征序号为（从0开始）\n"+str(self.feature_index))
        self.output_descriptor_name()

    def feature_selection2(self):
        '''
        递归特征消除法
        '''
        # model = SVR(kernel='linear')
        # model = BayesianRidge()
        # model = ElasticNet()
        # model = GradientBoostingRegressor()

        model = LinearRegression()
        rfe = RFE(estimator=model, n_features_to_select=20)
        rfe.fit(self.descriptor_tr, self.pIC50_tr)
        self.feature_index = rfe.get_support(True)
        print("递归特征消除法的特征序号为（从0开始）\n"+str(self.feature_index))
        self.output_descriptor_name()

    def feature_selection3(self):
        '''
        基于树模型的特征选择法
        '''
        gbdt = SelectFromModel(
            estimator=GradientBoostingRegressor())
        gbdt.fit(self.descriptor_tr, self.pIC50_tr)
        self.feature_index = gbdt.get_support(True)
        print("基于树模型的特征选择法的特征序号为（从0开始）\n"+str(self.feature_index))
        self.output_descriptor_name()

    def ERα_activity_dataset(self):
        '''
        生成ERα_activity_train.csv和ERα_activity_test.csv
        '''
        y_data_1_tr = np.transpose(self.IC50_nM_tr)
        y_data_2_tr = np.transpose(self.pIC50_tr)
        x_data_tr = dp.extract_matrix_column(
            self.descriptor_tr, self.feature_index)
        y_data_tr = dp.combine_matrix(y_data_1_tr, y_data_2_tr, orient='x')
        data_tr = dp.combine_matrix(y_data_tr, x_data_tr, orient='x')
        # print(data_tr)
        print(len(data_tr))
        print(len(data_tr[0]))
        path = self.pwd+"/ERα_activity_train.csv"
        print(str(path))
        dp.write_data(data_tr, path)
        x_data_te = dp.extract_matrix_column(
            self.descriptor_te, self.feature_index)
        # print(x_data_te)
        print(len(x_data_te))
        print(len(x_data_te[0]))
        path = self.pwd+"/ERα_activity_test.csv"
        print(str(path))
        dp.write_data(x_data_te, path)

    def ADMET_dataset(self, ADMET):
        '''
        ADMET=0,1,2,3,4
        生成ADMET_train[1-5].csv和ADMET_test[1-5].csv
        '''
        y_data_tr = dp.extract_matrix_column(self.ADMET_tr, [ADMET])
        x_data_tr = self.descriptor_tr
        data_tr = dp.combine_matrix(y_data_tr, x_data_tr, orient='x')
        # print(data_tr)
        print(len(data_tr))
        print(len(data_tr[0]))
        path = self.pwd+"/ADMET_train"+str(ADMET+1)+".csv"
        print(str(path))
        dp.write_data(data_tr, path)
        x_data_te = self.descriptor_te
        # print(x_data_te)
        print(len(x_data_te))
        print(len(x_data_te[0]))
        path = self.pwd+"/ADMET_test"+str(ADMET+1)+".csv"
        print(str(path))
        dp.write_data(x_data_te, path)

    def ERα_activity_tr_scaler(self):
        '''
        对ERα_activity.csv中的原始数据进行标准化
        '''
        data = dp.combine_matrix(self.IC50_nM_tr, self.pIC50_tr, orient='x')
        max = []
        max.append(np.max(self.IC50_nM_tr))
        max.append(np.max(self.pIC50_tr))
        min = []
        min.append(np.min(self.IC50_nM_tr))
        min.append(np.min(self.pIC50_tr))
        data = dp.scaler_data(data, max, min)
        path = self.pwd+"/ERα_activity_tr_scaler.csv"
        print(str(path))
        dp.write_data(data, path)

    def ERα_activity_tr_select(self):
        '''
        对ERα_activity.csv中的原始数据进行限制[4,9]
        '''
        data = []
        for _ in range(self.smiles_nb_tr):
            if self.pIC50_tr[_] >= 4 and self.pIC50_tr[_] <= 9:
                data.append([self.IC50_nM_tr[_], self.pIC50_tr[_]])
        path = self.pwd+"/ERα_activity_tr_select.csv"
        print(str(path))
        dp.write_data(data, path)


if __name__ == '__main__':
    smiles = SMILES()
    smiles.feature_selection1()
    smiles.ERα_activity_dataset()
    for _ in range(5):
        smiles.ADMET_dataset(_)
    # smiles.ERα_activity_tr_select()
