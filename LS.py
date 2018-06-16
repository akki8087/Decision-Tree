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


for co in count:
    p[co] = count[c]/total_count
    entropy += (-p[co]*math.log(p[co]))
    
def IG(data,k,entropy_dic):
    g = 0
    c = pd.crosstab(data['Loan_Status'],data[k], margins=True)
    for i in c.columns[:-1]:
        p1 = c[i]['Y']/c[i]['All']
        p2 = c[i]['N']/c[i]['All']
        w = c[i]['All']/c['All']['All']
        entropy_dic[i] = -p1*math.log(p1)-p2*math.log(p2)
        g += (w*entropy_dic[i])
    
    return g,entropy_dic
    

column = ['Gender', 'Married', 'Education', 'Self_Employed','Credit_History']
entropy1 = {}

ig = {}
for k in column:
    g,entropy1 = IG(X,k, entropy1)
    ig[k] = entropy - g
maxi = max(ig, key= ig.get)   

d1 = X.loc[X[maxi] == 1]
d2 = X.loc[X[maxi] == 0]
column2 = column.remove(maxi)

entropy3 = {}
ig3 = {}

for k in column:
    g3,entropy1 = IG(d1,k, entropy3)
    ig3[k] = entropy2[1] - g3
maxi1 = max(ig3, key= ig3.get)   

entropy2 = {}
g1,entropy2 = IG(X,maxi,entropy2)
ig2 = ig[maxi] - g1


