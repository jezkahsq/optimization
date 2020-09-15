# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 18:58:21 2020

@author: Administrator
"""


import sys
sys.path.append('C:\\Users\\Administrator\\Desktop\\参数求解\\MODEL')
import pandas as pd
import matplotlib.pyplot as plt
from GA_MODEL import GA_model 
from MODEL.GA_TEST import TEST
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] =False

data = pd.read_excel("C:\\Users\\Administrator\\Desktop\\参数求解\\参数求解数据\\1.xlsx")
size_pop = 70
max_iter = 500
prob_mut = 0.0001

for _ in range(20):
    print(f'第{_+1}次训练：')
    print(f'种群数量{size_pop}, 迭代次数{max_iter}, 变异概率{prob_mut}', '\n')
    
    My_model = GA_model(data, size_pop, max_iter, prob_mut)
    My_model.DataSelect()
    My_model.Model()

    C = My_model.best_x[0]
    N = My_model.best_x[1]
    A = My_model.best_x[2]

    My_test = TEST(data, C, N, A)
    My_test.Test()
