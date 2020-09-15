# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 17:13:31 2020

@author: Administrator
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class TEST:
    
    def __init__(self, data, C, N, A):
        self.data = data
        self.C = C
        self.N = N
        self.A = A
        self.t = self.data.iloc[:,0]
        self.h = self.data.iloc[:,1]
        self.waterin = self.data.iloc[:,5]
        self.flow = self.data.iloc[:,3]
        self.hz = self.data.iloc[:,4]

    def CoolingTower(self,dryball,humidity,inletwater,waterflow,Hz):
        PVSain=10**(2.7877+7.625*dryball/(241.6+dryball))
        PVain=PVSain*humidity/100
        Hain=1.006*dryball+(0.621955*PVain)/(101325-PVain)*(2501+1.83*dryball)
    
        PVSswi=10**(2.7877+7.625*inletwater/(241.6+inletwater))*1 #adjustcoef
        Hswi=1.006*inletwater+(0.621955*PVSswi)/(101325-PVSswi)*(2501+1.83*inletwater)
    
        outletwater=inletwater-5
        PVSswo=10**(2.7877+7.625*outletwater/(241.6+outletwater))
        Hswo=1.006*outletwater+(0.621955*PVSswo)/(101325-PVSswo)*(2501+1.83*outletwater)
    
        Cs=(Hswi-Hswo)/(inletwater-outletwater)
    
        ma=self.A*Hz/50 #to know 730 is a good value #455不同
        ma=ma/(0.0000136*36**2-0.0046675*36+1.29295)*(0.0000136*(inletwater-3)**2-0.0046675*(inletwater-3)+1.29295) #修正
        # ma adjusted coef, depend on density of air
        mw=waterflow
        Cpw=4.1868
        mstar=ma/mw*Cs/Cpw
    
        if mstar>1:
            mstar=1/mstar
    
        c=self.C
        n=self.N
        NTU=c*(ma/mw)**(-1-n)
        # adjusted coef for too much air
        sigmaa=(1-np.exp(-NTU*(1-mstar)))/(1-mstar*np.exp(-NTU*(1-mstar)))
        Qrej=sigmaa*ma*(Hswi-Hain)
        waterout=inletwater-Qrej/waterflow/4.1868
        return waterout
    
    def Test(self):
        self.mae = [0 for i in range(len(self.data))]
        for i in range(len(self.data)):
            self.result = self.CoolingTower(self.t[i], self.h[i], self.waterin[i], self.flow[i], self.hz[i])
            self.mae[i] = abs(self.result - self.data.iloc[i, 2])
        mae_sum = sum(self.mae)
        print('MAE:', mae_sum, '-'*70, '\n')
        plt.hist(self.mae, bins=100)
        plt.title('MAE频次分布直方图')
        plt.show()