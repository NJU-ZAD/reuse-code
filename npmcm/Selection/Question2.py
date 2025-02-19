#!/usr/bin/python3
# -*- coding:utf-8 -*-
import math
import os

import xlrd
from numpy import random


class BookB:
    '''
    B类图书模型
    '''

    def __init__(self, B, index):
        '''
        定义常量
        '''
        # 标识图书
        self.B = B
        # 标识子类书籍
        self.index = index
        # 初始化参数
        self.init_parameter()
        '''
        定义变量
        '''
        # 印刷的次数
        self.a = 0
        # 印刷的具体月份
        self.b = [0 for _ in range(7)]
        # 2021年每个月的销量
        self.p = [0 for _ in range(12)]
        # 2021年每个月的印刷量
        self.h = [0 for _ in range(12)]
        # 比例系数c_i
        self.c = [0 for _ in range(7)]
        # 随机变量d
        self.d = 0
        # 2021年每个月的库存
        self.w = [0 for _ in range(12)]
        # 目标函数值
        self.J = -10**9

    def read_xls(self):
        '''
        从xls文件读取数据
        '''
        pwd = os.getcwd()
        res = pwd.split('/')
        if res[-1] != "Selection":
            pwd += "/npmcm/Selection"
        xlsfile = pwd+"/BC_input_data.xls"
        book = xlrd.open_workbook(xlsfile)
        sheet = book.sheet_by_index(self.B-1)
        sheet_name = book.sheet_names()[self.B-1]
        print("正在读取"+str(sheet_name))
        # nrows = sheet.nrows
        # print("行总数="+str(nrows))
        # ncols = sheet.ncols
        # print("列总数="+str(ncols))
        if self.B == 1:
            cursor = 47
        elif self.B == 2:
            cursor = 42
        self.p_μ = [0 for _ in range(12)]
        self.p_σ = [0 for _ in range(12)]
        for i in range(12):
            σ = sheet.cell_value(cursor+i, 4*(self.index-1))
            μ = sheet.cell_value(cursor+i, 4*(self.index-1)+1)
            # print(μ, σ)
            self.p_μ[i] = μ
            self.p_σ[i] = σ

    def init_parameter(self):
        '''
        n表示销售折扣
        m表示图书定价
        r表示印刷印张
        k表示2021年以前的总销量
        _l表示2021年以前的总印刷量
        '''
        print("\n图书B"+str(self.B)+"-"+str(self.index))
        self.n = 0.18
        self.read_xls()
        if self.B == 1:
            if self.index == 1:
                self.m = 98.8
                self.r = 33
                self.k = 53928+73061+34105+15710
                self._l = 5000+30000+11000+11000+35000+20000 \
                    + 20000+10000+40000+10000+10000
            elif self.index == 2:
                self.m = 98.8
                self.r = 29.5
                self.k = 13490+20559+14490+7155
                self._l = 10000+15000+10000+10000+10000+5000 \
                    + 10000+8000+8600
            elif self.index == 3:
                self.m = 98.8
                self.r = 34
                self.k = 18763+25891+18002+9441
                self._l = 20000+10000+8200+8000+8200+20000+10000+5000
            elif self.index == 4:
                self.m = 98.8
                self.r = 31.5
                self.k = 8537+19786+8148+5206
                self._l = 5000+8000+5000+8000+6000+5000+10000+8000+10000
            elif self.index == 5:
                self.m = 98.8
                self.r = 26
                self.k = 4351+12129+6713+3067
                self._l = 5000+6000+6000+3000+10000+4000+10000+10000
        elif self.B == 2:
            if self.index == 1:
                self.m = 68.8
                self.r = 18.5
                self.k = 16681+58552+44003+33220
                self._l = 40000+15000+14000+20000+10000+10000 \
                    + 20000+12000+8000+16000+3000
            elif self.index == 2:
                self.m = 68.8
                self.r = 18.5
                self.k = 7910+21515+20980+13791
                self._l = 17000+10000+8000+15000+5000+10000 \
                    + 5000+5000+7000
            elif self.index == 3:
                self.m = 68.8
                self.r = 19.5
                self.k = 6445+26588+23855+11259
                self._l = 12000+5000+8000+6000+18000+2500 \
                    + 2500+10000+5000+4000+6000+10000+3000
            elif self.index == 4:
                self.m = 68.8
                self.r = 17.5
                self.k = 10260+19398+18190+12666
                self._l = 15000+11000+7000+10000+5000+5000 \
                    + 15000+10000+5000+5000+5000+8000
            elif self.index == 5:
                self.m = 68.8
                self.r = 17.5
                self.k = 1923+11464+8058+8755
                self._l = 8000+6000+3000+10000+5000+4000 \
                    + 4000+6000+10000

    def get_a(self):
        '''
        获得a的取值
        '''
        a = random.randint(1, 7+1)
        return a

    def get_b_i(self, a):
        '''
        获得b_i的取值
        '''
        b = [0 for _ in range(7)]
        for i in range(a):
            if self.B == 1:
                while True:
                    b[i] = random.randint(3, 9+1)
                    temp = b[0:i]
                    if b[i] not in temp:
                        break
            elif self.B == 2:
                while True:
                    b[i] = random.randint(10, 16+1)
                    if b[i] > 12:
                        b[i] -= 12
                    temp = b[0:i]
                    if b[i] not in temp:
                        break
        return b

    def get_c_i(self, a):
        '''
        获得c_i的取值
        '''
        c = [0 for _ in range(7)]
        for i in range(a):
            c[i] = random.rand()*(3-1)+1
        return c

    def get_d(self):
        '''
        获得d的取值
        '''
        while True:
            d = random.normal(loc=0, scale=100)
            if d > 0:
                break
        return d

    def get_p_i(self):
        '''
        随机生成符合正态分布的p_i
        '''
        p = [0 for _ in range(12)]
        for i in range(12):
            p[i] = random.normal(loc=self.p_μ[i], scale=self.p_σ[i])
        return p

    def f(self, x, μ, σ):
        '''
        正态分布的概率密度函数
        '''
        res = 1/(((2*math.pi)**0.5)*σ)*math.exp(-1*((x-μ)**2)/(2*(σ**2)))
        return res

    def get_index(self, b_i):
        '''
        得到具体月份的索引
        '''
        month1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        month2 = [4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3]
        if self.B == 1:
            return month1.index(b_i)
        elif self.B == 2:
            return month2.index(b_i)

    def get_h_b_N(self, a, b, c, d, p):
        '''
        计算印刷量h_b_N的值
        '''
        h = [0 for _ in range(12)]
        error = 0
        for i in range(a):
            _h = self._l
            for _ in range(12):
                if _ >= self.get_index(b[i]):
                    break
                _h += h[_]
            _p = self.k
            for _ in range(12):
                if _ > self.get_index(b[i]):
                    break
                _p += p[_]
            if _h-_p > 0:
                if _h-_p > d:
                    h[self.get_index(b[i])] = 0
                else:
                    h[self.get_index(b[i])] = c[i]*(_h-_p)
                    # print("h="+str(c[i]*(_h-_p)))
            else:
                error = 1
                break
        return (error, h)

    def get_s_b_N(self, h, b, N):
        '''
        根据印制月份b_N计算印制成本s_b_N
        '''
        res = 0
        if h[self.get_index(b[N-1])] >= 10000:
            res = 0.32*self.r*h[self.get_index(b[N-1])]
        else:
            res = 0.34*self.r*h[self.get_index(b[N-1])]
        return res

    def get_p_μ(self, i):
        '''
        得到p1+...+pi的均值
        '''
        res = 0
        for _ in range(i):
            res += self.p_μ[_]
        return res

    def get_p_σ(self, i):
        '''
        得到p1+...+pi的标准差
        '''
        res = 0
        for _ in range(i):
            res += ((self.p_σ[_])**2)
        res = res**0.5
        return res

    def get_y(self, a, b, p, h):
        '''
        根据p_i计算总的利润y
        '''
        total_p = 0
        for _ in range(12):
            total_p += p[_]
        total_s = 0
        for _ in range(a):
            total_s += self.get_s_b_N(h, b, _+1)
        res = self.m*self.n*total_p \
            - total_s-0.0273*self.m*total_p
        return res

    def get_w_i(self, i, p, h):
        '''
        根据p_i计算每月的库存量w_i
        '''
        total_p = 0
        for _ in range(i):
            total_p += p[_]
        total_h = 0
        for _ in range(i-1):
            total_h += h[_]
        res = self._l-self.k+total_h-total_p
        return res

    def slove(self):
        '''
        计算a、b_i和c_i的值
        '''
        for i in range(1000):
            a = self.get_a()
            b = self.get_b_i(a)
            c = self.get_c_i(a)
            d = self.get_d()
            J = 0
            for j in range(100):
                sum_p = 0
                sum_w = 0
                while True:
                    p = self.get_p_i()
                    (error, h) = self.get_h_b_N(a, b, c, d, p)
                    if error == 0:
                        break
                for k in range(12):
                    sum_p += p[k]
                    sum_p_i = 0
                    for _ in range(k+1):
                        sum_p_i += p[_]
                        z_i = 0.05*self.m*self.get_w_i(k+1, p, h)
                        sum_w += (z_i*self.f(sum_p_i, self.get_p_μ(k+1),
                                             self.get_p_σ(k+1)))
                J += (self.get_y(a, b, p, h) * self.f(sum_p,
                      self.get_p_μ(12), self.get_p_σ(12)))
                J -= sum_w
            # print("i="+str(i)+" J="+str(J))
            if J > self.J:
                self.a = a
                self.b = b
                self.c = c
                self.d = d
                self.p = p
                self.h = h
                self.J = J
        print("a的值为"+str(self.a))
        print("b的值为"+str(self.b))
        print("c的值为"+str(self.c))
        print("d的值为"+str(self.d))
        print("2021年每个月的销量p="+str(self.p))
        print("2021年每个月的印刷量h="+str(self.h))
        print("目标函数值J="+str(self.J))


if __name__ == '__main__':
    for i in range(2):
        for j in range(5):
            book = BookB(i+1, j+1)
            book.slove()
