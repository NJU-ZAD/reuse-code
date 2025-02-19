#!/usr/bin/python3
# -*- coding:utf-8 -*-
import math

from numpy import random


class BookA:
    '''
    A类图书模型
    '''

    def __init__(self, A):
        '''
        定义常量
        '''
        # 标识图书
        self.A = A
        # 第一次订单报数P(2021)
        self.p_μ = 0
        self.p_σ = 0
        self.p = 0
        # 总的订单报数G(2021)
        self.g_μ = 0
        self.g_σ = 0
        self.g = 0
        # 初始化参数
        self.init_parameter()

        '''
        定义变量
        '''
        # 参数a1，第一次印刷指标
        self.a = 0
        # 参数a2，第二次印刷指标
        self.b = 0
        # 参数a3，第三次印刷指标
        self.c = 0
        # 收益
        self.y = -10**9

    def init_parameter(self):
        '''
        n表示销售折扣
        m表示图书定价
        r表示印刷印张
        '''
        self.n = 0.48
        if self.A == 1:
            self.m = 12.57
            self.r = 8.5
            self.p_μ = 152942.722
            self.p_σ = 11565
            self.g_μ = 155338.967
            self.g_σ = 12216
        elif self.A == 2:
            self.m = 11.67
            self.r = 8.5
            self.p_μ = 135030.660
            self.p_σ = 17551
            self.g_μ = 146005.355
            self.g_σ = 18603
        elif self.A == 3:
            self.m = 11.65
            self.r = 8.5
            self.p_μ = 21617.736
            self.p_σ = 6536
            self.g_μ = 23332.274
            self.g_σ = 7875
        elif self.A == 4:
            self.m = 10.6
            self.r = 7.5
            self.p_μ = 162757
            self.p_σ = 15045
            self.g_μ = 172292
            self.g_σ = 15641
        elif self.A == 5:
            self.m = 10.9
            self.r = 7.5
            self.p_μ = 178699.094
            self.p_σ = 23797
            self.g_μ = 184838.600
            self.g_σ = 26113

    def get_pg(self):
        '''
        随机生成符合正态分布的p和g
        '''
        p = random.normal(loc=self.p_μ, scale=self.p_σ)
        g = random.normal(loc=self.g_μ, scale=self.g_σ)
        return (p, g)

    def f(self, x, μ, σ):
        '''
        正态分布的概率密度函数
        '''
        res = 1/(((2*math.pi)**0.5)*σ)*math.exp(-1*((x-μ)**2)/(2*(σ**2)))
        return res

    def get_abc(self):
        '''
        生成随机的印刷指标a、b、c
        a 1-2
        b 0-0.7
        c 0-0.3
        '''
        a = random.rand()*(2-1)+1
        b = random.rand()*(0.7-0)+0
        c = random.rand()*(0.3-0)+0
        return (a, b, c)

    def get_s(self, a_i, p):
        '''
        根据印制数量a_i*p计算印制成本s
        '''
        res = 0
        if a_i*p >= 10000:
            res = 0.35*a_i*p*self.r
        else:
            res = 0.41*a_i*p*self.r
        return res

    def get_y(self, a, b, c, p, g):
        '''
        根据a_i、p、g计算收益y
        '''
        res = self.m*self.n*g \
            - (self.get_s(a, p)+self.get_s(b, p) + self.get_s(c, p)) \
            - 0.0273*self.m*g
        return res

    def slove(self):
        '''
        计算印刷指标a、b、c
        '''
        for i in range(10000):
            (a, b, c) = self.get_abc()
            y = 0
            for j in range(500):
                while True:
                    (p, g) = self.get_pg()
                    if p <= g and (a+b+c)*p >= g:
                        break
                y += (self.get_y(a, b, c, p, g) *
                      self.f(p, self.p_μ, self.p_σ))
            if y > self.y:
                self.y = y
                self.a = a
                self.b = b
                self.c = c
                self.p = p
                self.g = g
        print("A类图书"+str(self.A))
        print("a1的值为a="+str(self.a))
        print("a2的值为b="+str(self.b))
        print("a3的值为c="+str(self.c))
        print("2021年第一次订单报数p="+str(self.p))
        print("2021年总的订单报数g="+str(self.g))
        print("最大收益y="+str(self.y))


if __name__ == '__main__':
    for _ in range(5):
        book = BookA(_+1)
        book.slove()
