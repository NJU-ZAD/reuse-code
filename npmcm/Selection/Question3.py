#!/usr/bin/python3
# -*- coding:utf-8 -*-
import math
import os

import xlrd
from numpy import random


class BookC:
    '''
    C类图书模型
    '''

    def __init__(self, C):
        '''
        定义常量
        '''
        # 标识图书
        self.C = C
        # 初始化参数
        self.init_parameter()
        '''
        定义变量
        '''
        # 随机变量b
        self.b = 0
        # 变动系数c
        self.c = 0
        # 每个月的热度指标
        self.d = [0 for _ in range(12)]
        # 每个月的销量
        self.p = [0 for _ in range(12)]
        # 每个月的库存
        self.w = [0 for _ in range(12)]
        # 每个月的印刷量
        self.h = [0 for _ in range(12)]
        # 每个月的印制成本
        self.s = [0 for _ in range(12)]
        # 总的利润
        self.y = -10**9

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
        if 1 <= self.C <= 3:
            sheet_id = 2
        elif 4 <= self.C <= 6:
            sheet_id = 3
        elif 7 <= self.C <= 9:
            sheet_id = 4
        sheet = book.sheet_by_index(sheet_id)
        sheet_name = book.sheet_names()[sheet_id]
        print("正在读取"+str(sheet_name))
        # nrows = sheet.nrows
        # print("行总数="+str(nrows))
        # ncols = sheet.ncols
        # print("列总数="+str(ncols))
        if self.C == 1:
            cursor_x = 0
            cursor_y = 37
        elif self.C == 2:
            cursor_x = 4
            cursor_y = 37
        elif self.C == 3:
            cursor_x = 8
            cursor_y = 37
        elif self.C == 4:
            cursor_x = 0
            cursor_y = 28
        elif self.C == 5:
            cursor_x = 4
            cursor_y = 29
        elif self.C == 6:
            cursor_x = 8
            cursor_y = 28
        elif self.C == 7:
            cursor_x = 0
            cursor_y = 58
        elif self.C == 8:
            cursor_x = 4
            cursor_y = 58
        elif self.C == 9:
            cursor_x = 8
            cursor_y = 58
        self.p_μ = [0 for _ in range(12)]
        self.p_σ = [0 for _ in range(12)]
        for i in range(12):
            μ = sheet.cell_value(cursor_y+i, cursor_x+1)
            up = sheet.cell_value(cursor_y+i, cursor_x+3)
            self.p_μ[i] = μ
            self.p_σ[i] = (up-μ)/1.96
            # print(self.p_σ[i], self.p_μ[i], up)

    def init_parameter(self):
        '''
        n表示销售折扣
        m表示图书定价
        r表示印刷印张
        color表示正文色数单色(1)双色(2)四色(4)
        k表示2021年3月31日以前的总销量
        _l表示2021年3月31日以前的总印刷量
        '''
        print("\n图书C"+str(self.C)+"类")
        self.n = 0.45
        self.read_xls()
        if self.C == 1:
            self.m = 28.80
            self.r = 10
            self.color = 2
            self.k = 5132
            self._l = 2*5000
        elif self.C == 2:
            self.m = 28.80
            self.r = 11
            self.color = 2
            self.k = 7299
            self._l = 2*5000
        elif self.C == 3:
            self.m = 68.00
            self.r = 31
            self.color = 1
            self.k = 15862
            self._l = 4*5000
        elif self.C == 4:
            self.m = 48.00
            self.r = 10.5
            self.color = 1
            self.k = 6859
            self._l = 5000+3000+3000
        elif self.C == 5:
            self.m = 30.00
            self.r = 10
            self.color = 1
            self.k = 9785
            self._l = 5000+3000+5000
        elif self.C == 6:
            self.m = 30.00
            self.r = 10
            self.color = 1
            self.k = 4200
            self._l = 5000+5000
        elif self.C == 7:
            self.m = 32.00
            self.r = 6
            self.color = 4
            self.k = 55865
            self._l = 10000+10000+5000+10000+9860+20000+6060
        elif self.C == 8:
            self.m = 24.00
            self.r = 6
            self.color = 4
            self.k = 57411
            self._l = 10000+10000+15000+10000+20000+10000
        elif self.C == 9:
            self.m = 38.00
            self.r = 9.5
            self.color = 4
            self.k = 18453
            self._l = 5000+5000+5000+10000

    def get_b(self):
        '''
        获得常数b
        '''
        b = random.randint(1, 1000+1)
        return b

    def get_c(self):
        '''
        获得c的取值
        '''
        c = random.rand()*(3-1)+1
        return c

    def get_d_i(self, i):
        '''
        获得d_i的取值
        '''
        total_μ = 0
        for _ in range(12):
            if _+1 >= i:
                total_μ += self.p_μ[_]
        if 1 <= i <= 11:
            d = (total_μ-self.p_μ[i-1])/total_μ
        elif i == 12:
            d = 0
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

    def get_w_i(self, p):
        '''
        计算每个月的库存量
        '''
        w = [0 for _ in range(12)]
        error = 0
        for i in range(12):
            total_p = 0
            for _ in range(i):
                total_p += p[_]
            w[i] = self._l-self.k-total_p
            if w[i] < 0:
                error = 1
                break
        return (error, w)

    def get_h(self, b, c, w, p):
        '''
        计算印刷量h的值
        '''
        h = [0 for _ in range(12)]
        for i in range(12):
            if w[i] - p[i] < b:
                h[i] = c*w[i]
        return h

    def get_s(self, h):
        '''
        根据印制量h计算印制成本s
        '''
        res = 0
        for i in range(12):
            h_i = h[i]
            if self.color == 1:
                if h_i >= 10000:
                    s = 0.32*self.r*h_i
                else:
                    s = 0.34*self.r*h_i
            elif self.color == 2:
                if h_i >= 10000:
                    s = 0.35*self.r*h_i
                else:
                    s = 0.41*self.r*h_i
            elif self.color == 4:
                if h_i >= 10000:
                    s = 0.45*self.r*h_i
                else:
                    s = 0.54*self.r*h_i
            res += s
        return res

    def get_p_μ(self):
        '''
        得到p*的均值
        '''
        res = 0
        for _ in range(12):
            res += self.p_μ[_]
        return res

    def get_p_σ(self):
        '''
        得到p*的标准差
        '''
        res = 0
        for _ in range(12):
            res += ((self.p_σ[_])**2)
        res = res**0.5
        return res

    def get_y(self, d, s, p):
        '''
        根据p_i计算总的利润y
        '''
        total_p = 0
        for _ in range(12):
            total_p += p[_]
        res = (1+d)*self.m*self.n*total_p \
            - s - 0.0273*self.m*total_p
        return res

    def slove(self):
        '''
        计算c的值
        '''
        d = self.get_d_i(1)
        for i in range(2000):
            b = self.get_b()
            c = self.get_c()
            y = 0
            for j in range(500):
                while True:
                    p = self.get_p_i()
                    (error, w) = self.get_w_i(p)
                    if error == 0:
                        break
                sum_p = 0
                for k in range(12):
                    sum_p += p[k]
                h = self.get_h(b, c, w, p)
                s = self.get_s(h)
                y += (self.get_y(d, s, p) * self.f(sum_p,
                      self.get_p_μ(), self.get_p_σ()))
            # print("i="+str(i)+" y="+str(y))
            if y > self.y:
                self.b = b
                self.c = c
                self.d = d
                self.p = p
                self.h = h
                self.w = w
                self.s = s
                self.y = y
        print("b的值为"+str(self.b))
        print("c的值为"+str(self.c))
        print("d1的值为"+str(self.d))
        print("p的值为"+str(self.p))
        print("h的值为"+str(self.h))
        print("w的值为"+str(self.w))
        print("s的值为"+str(self.s))
        print("总的利润y="+str(self.y))


if __name__ == '__main__':
    for i in range(9):
        book = BookC(i+1)
        book.slove()
