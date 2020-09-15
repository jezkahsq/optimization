# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 10:19:47 2020

@author: Administrator
"""


import numpy as np
from sko.GA import GA
import pandas as pd
import matplotlib.pyplot as plt
import time


class GA_model:
    
    def __init__(self, data, size_pop, max_iter, prob_mut):
        self.data = data
        self.size_pop = size_pop
        self.max_iter = max_iter
        self.prob_mut = prob_mut
    
    
    def DataSelect(self):
        b = self.data.iloc[:,2].max()
        a = self.data.iloc[:,2].min()
        abrange = np.linspace(a, b, 11)
        data1 = self.data[(self.data.iloc[:,2]<abrange[1]) & (self.data.iloc[:,2]>=abrange[0])]
        data2 = self.data[(self.data.iloc[:,2]<abrange[2]) & (self.data.iloc[:,2]>=abrange[1])]
        data3 = self.data[(self.data.iloc[:,2]<abrange[3]) & (self.data.iloc[:,2]>=abrange[2])]
        data4 = self.data[(self.data.iloc[:,2]<abrange[4]) & (self.data.iloc[:,2]>=abrange[3])]
        data5 = self.data[(self.data.iloc[:,2]<abrange[5]) & (self.data.iloc[:,2]>=abrange[4])]
        data6 = self.data[(self.data.iloc[:,2]<abrange[6]) & (self.data.iloc[:,2]>=abrange[5])]
        data7 = self.data[(self.data.iloc[:,2]<abrange[7]) & (self.data.iloc[:,2]>=abrange[6])]
        data8 = self.data[(self.data.iloc[:,2]<abrange[8]) & (self.data.iloc[:,2]>=abrange[7])]
        data9 = self.data[(self.data.iloc[:,2]<abrange[9]) & (self.data.iloc[:,2]>=abrange[8])]
        data10 = self.data[(self.data.iloc[:,2]<abrange[10]) & (self.data.iloc[:,2]>=abrange[9])]
        data1 = data1.sample(frac=0.03, replace=False, axis=0)
        data2 = data2.sample(frac=0.03, replace=False, axis=0)
        data3 = data3.sample(frac=0.03, replace=False, axis=0)
        data4 = data4.sample(frac=0.03, replace=False, axis=0)
        data5 = data5.sample(frac=0.03, replace=False, axis=0)
        data6 = data6.sample(frac=0.03, replace=False, axis=0)
        data7 = data7.sample(frac=0.03, replace=False, axis=0)
        data8 = data8.sample(frac=0.03, replace=False, axis=0)
        data9 = data9.sample(frac=0.03, replace=False, axis=0)
        data10 = data10.sample(frac=0.03, replace=False, axis=0)
        frames = [data1, data2, data3, data4, data5, data6, data7, data8, data9, data10]
        self.data_ = pd.concat(frames)
        self.data_ = self.data_.reset_index(drop=True)
        self.t = self.data_.iloc[:,0]
        self.h = self.data_.iloc[:,1]
        self.waterout = self.data_.iloc[:,2]
        self.flow = self.data_.iloc[:,3]
        self.hz = self.data_.iloc[:,4]
        self.waterin = self.data_.iloc[:,5]
        
    
    def CoolingTower(self,dryball,humidity,inletwater,waterflow,Hz,outletwater,c,n,a):
        PVSain=10**(2.7877+7.625*dryball/(241.6+dryball))
        PVain=PVSain*humidity/100
        Hain=1.006*dryball+(0.621955*PVain)/(101325-PVain)*(2501+1.83*dryball)
    
        PVSswi=10**(2.7877+7.625*inletwater/(241.6+inletwater))*1 #adjustcoef
        Hswi=1.006*inletwater+(0.621955*PVSswi)/(101325-PVSswi)*(2501+1.83*inletwater)
    
        outletwater=inletwater-5
        PVSswo=10**(2.7877+7.625*outletwater/(241.6+outletwater))
        Hswo=1.006*outletwater+(0.621955*PVSswo)/(101325-PVSswo)*(2501+1.83*outletwater)
    
        Cs=(Hswi-Hswo)/(inletwater-outletwater)
    
        ma=a*Hz/50 #to know 730 is a good value 
        ma=ma/(0.0000136*36**2-0.0046675*36+1.29295)*(0.0000136*(inletwater-3)**2-0.0046675*(inletwater-3)+1.29295) #修正
        # ma adjusted coef, depend on density of air
        mw=waterflow
        Cpw=4.1868
        mstar=ma/mw*Cs/Cpw
    
        if mstar>1:
            mstar=1/mstar
    
        NTU=c*(ma/mw)**(-1-n)
        # adjusted coef for too much air
        sigmaa=(1-np.exp(-NTU*(1-mstar)))/(1-mstar*np.exp(-NTU*(1-mstar)))
        Qrej=sigmaa*ma*(Hswi-Hain)
        return abs(inletwater-Qrej/waterflow/4.1868-outletwater)

#%%
    # def ObjectFunction(self,p):
    #     x1, x2 ,x3 = p
    #     res = [0 for i in range(len(self.t))]
    #     for i in range(len(self.t)):
    #         res[i] = self.CoolingTower(self.t[i],self.h[i],self.waterin[i],self.flow[i],self.hz[i],self.waterout[i],*p)
    #     self.object_function = sum(res)
    #     return sum(res)


    # def Model(self):
    #     time_start=time.time()
    #     ga = GA(func=self.ObjectFunction, n_dim=3, size_pop=70, max_iter=10, prob_mut=0.0001, lb=[-3, -3, 600], ub=[3, 3, 1000], precision=1e-5)
    #     # 默认变异率prob_mut0.001
    #     best_x, best_y = ga.run()
    #     print('best_x:', best_x, '\n', 'best_y:', best_y)
    #     time_end=time.time()
    #     print('time cost of GA',time_end-time_start,'s', '-'*70)

    #     Y_history = pd.DataFrame(ga.all_history_Y)
    #     fig, ax = plt.subplots(2, 1)
    #     ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
    #     Y_history.min(axis=1).cummin().plot(kind='line')
    #     plt.show()
    
#%%
    def Model(self):
        def ObjectFunction(p):
            x1, x2 ,x3 = p
            res = [0 for i in range(len(self.t))]
            for i in range(len(self.t)):
                res[i] = self.CoolingTower(self.t[i],self.h[i],self.waterin[i],self.flow[i],self.hz[i],self.waterout[i],*p)
            return sum(res)

        time_start=time.time()
        ga = GA(func=ObjectFunction, n_dim=3, size_pop=self.size_pop, max_iter=self.max_iter, prob_mut=self.prob_mut, lb=[-3, -3, 600], ub=[3, 3, 1000], precision=1e-5)
        # 默认变异率prob_mut0.001
        self.best_x, self.best_y = ga.run()
        print('best_x:', self.best_x, '\n', 'best_y:', self.best_y)
        time_end=time.time()
        print('time cost of GA',time_end-time_start,'s')

        Y_history = pd.DataFrame(ga.all_history_Y)
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
        Y_history.min(axis=1).cummin().plot(kind='line')
        plt.show()
    