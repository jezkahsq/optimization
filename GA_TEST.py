# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 17:13:31 2020

@author: Administrator
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sci
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore") 

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
        self.waterout=inletwater-Qrej/waterflow/4.1868
        return self.waterout
    
        
    def Test(self):
        self.mae = [0 for i in range(len(self.data))]
        self.result = [0 for i in range(len(self.data))]
        for i in range(len(self.data)):
            self.result[i] = self.CoolingTower(self.t[i], self.h[i], self.waterin[i], self.flow[i], self.hz[i])
            self.mae[i] = abs(self.result[i] - self.data.iloc[i, 2])
        mae_sum = sum(self.mae) / len(self.data)
        # print('MAE:', mae_sum)
        plt.hist(self.mae, bins=100)
        plt.title(f'参数{self.C} {self.N} {self.A}MAE频次分布')
        plt.show()
        
        
    def EvaluationIndex(self):                                                               
        model_metrics_functions = [explained_variance_score, mean_absolute_error, mean_squared_error, r2_score]
        model_metrics_list = [m(self.result, self.data.iloc[:,2]) for m in model_metrics_functions]  #回归评估指标列表
        # self.regression_score = pd.DataFrame(model_metrics_list, columns=['explained_variance', 'mae', 'mse', 'r2']) #建立回归指标的数据框
#        print('all samples: %d \t features: %d' % (self.n_samples, self.n_features), '\n', '-'*60)
        print('\n', 'explained_variance      mae                mse                 r2')
        print(model_metrics_list,'-'*70)
        
        
    def EvaluationPlot(self):
        plt.plot(np.arange(len(self.result)), self.result, 'g--')
        plt.plot(np.arange(len(self.data.iloc[:,2])), self.data.iloc[:,2], 'k-', label = 'true')
        plt.legend(loc='upper right')
        plt.title(f'参数{self.C} {self.N} {self.A}预测值真实值折线图')
        plt.show()
        sns.jointplot(x=self.result, y=self.data.iloc[:,2], kind = 'hex', color = 'k', stat_func=sci.pearsonr,
                      marginal_kws = dict(bins = 20))
        plt.show()
        
        
    def RelativeErrorSorted(self):
        self.mae = pd.DataFrame(self.mae)
        self.result = pd.DataFrame(self.result)
        frames = [self.mae, self.result]
        self.mae_pre = pd.concat(frames, axis=1, names=['mae', 'pre'])
        print(self.mae_pre)
        a = self.data.iloc[:,2].max()
        b = self.data.iloc[:,2].min()
        abrange = np.linspace(b, a, 11)
        self.y_test =  self.data.iloc[:,2]

        for j in range(0, 10):
            plt.hist(self.mae_pre.iloc[:,0][(self.mae_pre.iloc[:,1]>=abrange[j]) & (self.mae_pre.iloc[:,1]<abrange[j+1])], bins=100, alpha=0.5, histtype='stepfilled', color='steelblue')
            #plt.xlim(-0.5, 0.5)
            down = abrange[j]
            up = abrange[j+1]
            plt.title(f'[{down},{up}]')
            plt.tick_params(labelsize=9)
            plt.savefig(f'C:\\Users\\Administrator\\Desktop\\参数求解\\figures\\'+str(f'参数{self.C} {self.N} {self.A}-[{down},{up}]')+'.png', dpi=300)
            plt.show()
        # plt.show()