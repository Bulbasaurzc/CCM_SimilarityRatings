#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 16:54:56 2021

@author: dawooood
Usage 

python3 model_hum_corr.py path_to_model_features path_to_avg_human_ratings

"""
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate



def calc_corr(model,human_mat):
    hum_sim_surv = pd.read_csv(human_mat,header=None)
    hum_avg_ratings = hum_sim_surv.loc[0].to_numpy()
    
    hum_sim_mat = np.zeros((18,18))
    ind = 0
    hum_sim_mat[:] = np.nan
    for i in range(18):
        for j in range(i,18):
            if i!=j:
                hum_sim_mat[i,j] = hum_avg_ratings[ind]
                hum_sim_mat[j,i] = hum_avg_ratings[ind]
                ind+=1
            else:
                hum_sim_mat[i,j] = 6
    hum_sim_mat_corr = np.reshape(hum_sim_mat,(18*18))
    F =np.genfromtxt(model,delimiter=',')
    #F = np.reshape(F, (18,4096))
    
    model_sim = np.dot(F,np.transpose(F))
    model_sim_corr =  np.reshape(model_sim, (18*18))
    
    deep_corr = (np.corrcoef(hum_sim_mat_corr,model_sim_corr)[0,1])**2
    
    hum_sim_i_j = []
    for i in range(hum_sim_mat.shape[0]):
        for j in range(i,hum_sim_mat.shape[1]):
            hum_sim_i_j.append(hum_sim_mat[i,j])
            
    model_F= []
    for i in range(F.shape[0]):
        for j in range(i,F.shape[0]):
            model_F.append(F[i] * F[j])
            
    model_F_cv= []
    for i in range(F.shape[0]):
        for j in range(F.shape[0]):
            model_F_cv.append(F[i] * F[j])    
    
    reg = Ridge(solver='sag', fit_intercept=False)
    parameters = {'alpha': [10,100,1000,1e4, 50000, 1e5,1e6]}
    search = GridSearchCV(reg, parameters, scoring='neg_mean_squared_error', cv=6)
    search.fit(model_F, hum_sim_i_j)
    
    best_reg = search.best_estimator_
    #print(best_reg)
    
    
    a=cross_validate(best_reg,model_F_cv,hum_sim_mat_corr,scoring="r2",cv=6)
    
    PredSimMat = best_reg.predict(model_F_cv)

    cor_mat = np.corrcoef(PredSimMat, hum_sim_mat.reshape(18*18))
    r = cor_mat[0,1]

    r2 = r**2
    
    adap_corr = r2
    
    return deep_corr,adap_corr
        

    
if __name__ == "__main__":
    model = sys.argv[1]
    human_mat = sys.argv[2]
    deep_corr, adap_corr = calc_corr(model,human_mat)
    print(f'Deep representation : correlation  = {round(deep_corr,2)}')
    print(f'Adapting representation : correlation = {round(adap_corr,2)}')
    