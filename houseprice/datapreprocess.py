# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 09:37:55 2018
## this is data preprocess for kaggle/houseprice
@author: zxcao
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.stats import skew

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--orgdata',default='train.csv')
parser.add_argument('--outdata',default='train.npz')
parser.add_argument('--label',default='SalePrice')
parser.add_argument('--id',default='Id')

def load_data(org_data,out_data,label,ID):
    
     orgdf = pd.read_csv(org_data)
     
     # 删除无用特征
     del orgdf['Utilities']
     
     # 删除测试集中没有的特征
     dellists = ['Condition2','HouseStyle','RoofMatl','Exterior1st','Exterior2nd',
                 'Heating','Electrical','MiscFeature']
     for fea in dellists:
        del orgdf[fea]
     
     y = np.array([])
    
     if label in orgdf.columns:
         
         Xdf,y,org_id = orgdf,orgdf.pop(label),orgdf.pop(ID)
         y = np.log1p(list(y))
            
     else:
         
         Xdf,org_id = orgdf,orgdf.pop(ID)
    
     del orgdf
     
     
     cols = Xdf.columns
     # 删除训练数据中的异常数据
     if label in cols:
         Xdf = Xdf.drop(Xdf[(Xdf['GrLivArea']>4000) & (Xdf[label]<300000)].index)
         

     # missing data fillna
     ## 1-fill none
     fillnons = ['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','GarageCond',
                 'GarageFinish','GarageQual','GarageType','BsmtCond','BsmtExposure',
                 'BsmtQual','BsmtFinType2','BsmtFinType1','MasVnrType']
     for col in fillnons:
        if col in cols:
            Xdf[col] = Xdf[col].fillna('None')
     
     del fillnons
     
     ## 2- fill median
     Xdf['LotFrontage'] =Xdf.groupby('Neighborhood')['LotFrontage'].transform(
             lambda x: x.fillna(x.median()))
     
     ## 3- fill 0
     fillos=['GarageYrBlt','MasVnrArea','BsmtFullBath','BsmtHalfBath','BsmtFinSF1',
             'BsmtFinSF2','BsmtUnfSF','GarageArea','GarageCars','TotalBsmtSF']
     
     for col in fillos:
         if col in cols:
            Xdf[col] = Xdf[col].fillna(0)
         
     del fillos
     
     ## 4- fill mode
     fillmds=['MSZoning','Functional','Electrical','Exterior1st','Exterior2nd','KitchenQual',
              'SaleType']
     for col in fillmds:
        if col in cols:
            Xdf[col] =Xdf[col].fillna(Xdf[col].mode()[0])
     
     del fillmds
     
     # numerical 2 categorical
     n2cCols =['MSSubClass','OverallCond','YrSold','MoSold']
     for col in n2cCols:
         if col in cols:
            Xdf[col] = Xdf[col].astype(str)
     del n2cCols
    
     # categorical 2 numerical
     
     c2nCols=['FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold']
     
     for col in c2nCols:
            if col in cols:
                lbl =LabelEncoder()
                lbl.fit(list(Xdf[col].values))
                Xdf[col] = lbl.transform(list(Xdf[col].values))
         
     del c2nCols
     
     # log numerical
     
     logncols=Xdf.dtypes[Xdf.dtypes!='object'].index
     skew_feats = Xdf[logncols].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
     skewness = pd.DataFrame({'Skew':skew_feats})
     skewness = skewness[abs(skewness)>0.75]
     
     for col in skewness.index:
         Xdf[col] = np.log1p(Xdf[col])
     
     # categorical onehot
     Xdf = pd.get_dummies(Xdf)
     cols_new = Xdf.columns
     X = np.array(Xdf)
     del cols
     np.savez(out_data,X=X,y=y,org_id=org_id,cols=cols_new)
     
     return

def main():
    args = parser.parse_args()
    
    org_data = args.orgdata
    out_data = args.outdata
    label =args.label
    ID = args.id
    
    load_data(org_data,out_data,label,ID)
    
if __name__=='__main__':
    main()
    
    
     
         
         
     
    
    
    