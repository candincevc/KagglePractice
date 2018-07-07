# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 20:10:45 2018
thi is the model file for house price
@author: maomao
"""

import pandas as pd
import numpy as np

from sklearn.linear_model import ElasticNet, Lasso
from sklearn.ensemble import   GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score

import argparse
parser =argparse.ArgumentParser()

parser.add_argument('--train',default='train.npz')
parser.add_argument('--test',default='test.npz')

# load data
def load_data(fileName):
    ''' load dataset '''
    files = np.load(fileName)
    X = files['X']
    y = files['y']
    org_id =files['org_id']
    
    return X,y,org_id



def rmsle_cv(model,X_train,y_train):
    kf = KFold(n_splits=5, shuffle=True, random_state=42).get_n_splits(X_train)
    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = kf))
    
    return(rmse)

# average model- model stacking1
    
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self,models):
        self.models = models
    
    def fit(self,X,y):
        self.models_ =[clone(x) for x in self.models]
        
        # train the dataset
        for model in self.models_:
             model.fit(X,y)
        
        return self
    
    def predict(self,X):
        predictions=np.column_stack([model.predict(X) for model in self.models_])
        return np.mean(predictions,axis=1)

# stacking averaged Models
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self,base_models,meta_model,n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
    
    def fit(self,X,y):
         self.base_models_ = [list() for x in self.base_models]
         self.meta_model_ = clone(self.meta_model)
         kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
          
         out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
         
         for i,model in enumerate(self.base_models):
             for train_index,test_index in kfold.split(X,y):
                 instance =clone(model)
                 self.base_models_[i].append(instance) # 保存每一折交叉验证的模型
                 instance.fit(X[train_index],y[train_index])
                 y_pred = instance.predict(X[test_index]) # 每一折交叉验证的结果
                 out_of_fold_predictions[test_index,i] = y_pred
         
         self.meta_model.fit(out_of_fold_predictions,y)
         
         return self
     
    def predict(self,X):
        
        meta_features = np.column_stack([
                np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
        for base_models in self.base_models_])
        
        return self.meta_model_.predict(meta_features)



#  model test
def selectModels(X_train,y_train,models):
    
    score =10000.0
    j=0
    for i, model in enumerate(models):
        rmsle=rmsle_cv(model,X_train,y_train)
        print('\n scores of model {}:{:.4f}({:.4f})\n'.format(i,rmsle.mean(),rmsle.std()))
        if rmsle.mean() <score:
            score = rmsle.mean()
            j=i
    return models[j]
        


def getSub(X_test,models,test_id):
    y_pred=models.predict(X_test)
    y_pred=np.expm1(y_pred)
    sub=pd.DataFrame()
    sub['Id'] = test_id
    sub['SalePrice'] = y_pred
    sub.to_csv(r'F:\GitHub\KagglePractice\houseprice\submission.csv',index=False)
    return
    
def main():
    args = parser.parse_args()
    trainpath =args.train
    testpath = args.test
    X_train,y_train,train_id =load_data(trainpath)
    X_test,y_test,test_id = load_data(testpath)
    
    # construct model
    # construct base model
    lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
    ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
    KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
    GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
    
    averaged_models =AveragingModels(models=(ENet,GBoost,KRR,lasso))
    stacked_averaged_models=StackingAveragedModels(base_models=(ENet, GBoost, KRR),meta_model=lasso)       
    models=(lasso,ENet,KRR,GBoost,averaged_models,stacked_averaged_models)
    
    # select model
    model_final =selectModels(X_train,y_train,models)
    getSub(X_test,model_final,test_id)
    
if __name__=='__main__':
    main()
    
    
