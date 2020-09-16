# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 10:53:23 2020

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
import warnings
warnings.filterwarnings("ignore") 

data = pd.read_excel("C:\\Users\\Administrator\\Desktop\\参数求解\\参数求解数据\\1.xlsx")
parameter = pd.read_excel("C:\\Users\\Administrator\\Desktop\\参数求解\\测试结果.xlsx", names=['c','n','a'])

for _ in range(0, len(parameter)):
    # print(_+2)
    C = parameter.iloc[_, 0]
    N = parameter.iloc[_, 1]
    A = parameter.iloc[_, 2]
    print(f'参数c {C} n {N} a {A}')
    My_test = TEST(data, C, N, A)
    My_test.Test()
    My_test.EvaluationIndex()
    My_test.EvaluationPlot()
    # My_test.RelativeErrorSorted()