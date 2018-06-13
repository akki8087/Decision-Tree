# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 00:36:23 2018

@author: NP
"""

import numpy as np
import pandas as pd
import math

data = pd.read_csv('loantrain.csv')

X = data.iloc[:,1:6] 

X['Loan_Status'] = data['Loan_Status']
X['Credit_History'] = data['Credit_History']

count = dict(X['Loan_Status'].value_counts())

total_count = len(X)

p = {}
entropy = 0


for c in count:
    p[c] = count[c]/total_count
    entropy += (-p[c]*math.log(p[c]))
    
def IG(k,entropy_dic):
    g = 0
    c = pd.crosstab(X['Loan_Status'],X[k], margins=True)
    for i in c.columns[:-1]:
        p1 = c[i]['Y']/c[i]['All']
        p2 = c[i]['N']/c[i]['All']
        w = c[i]['All']/c['All']['All']
        entropy_dic[i] = -p1*math.log(p1)-p2*math.log(p2)
        g = (w*entropy_dic[i])
    
    return g
    

column = ['Gender', 'Married', 'Education', 'Self_Employed','Credit_History']
entropy1 = {}

ig = {}
for k in column:
    ig[k] = entropy - IG(k, entropy1)
maxi = max(ig, key= ig.get)   
entropy2 = {}

ig2 = ig[maxi] - IG(maxi,entropy2)


