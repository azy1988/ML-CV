# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 22:05:39 2018

@author: Administrator
"""

# import libraries necessary for this project
import os, sys, pickle
 
import numpy as np
import pandas as pd
 
from datetime import date
 
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, roc_auc_score, auc, roc_curve
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier


# Convert Discount_rate and Distance
def getDiscountType(row):
    if row == 'null':
        return 'null'
    elif ':' in row:
        return 1
    else:
        return 0

def convertRate(row):
    """Convert discount to rate"""
    if row == 'null':
        return 1.0
    elif ':' in row:
        rows = row.split(':')
        return 1.0 - float(rows[1])/float(rows[0])
    else:
        return float(row)
    
def getDiscountMan(row):
    if ':' in row:
        rows = row.split(':')
        return int(rows[0])
    else:
        return 0

def getDiscountJian(row):
    if ':' in row:
        rows = row.split(':')
        return int(rows[1])
    else:
        return 0
    
def processData(df):
    
    # convert discount_rate
    df['discount_type'] = df['Discount_rate'].apply(getDiscountType)
    df['discount_rate'] = df['Discount_rate'].apply(convertRate)
    df['discount_man'] = df['Discount_rate'].apply(getDiscountMan)
    df['discount_jian'] = df['Discount_rate'].apply(getDiscountJian)
    
    print(df['discount_rate'].unique())
    
    return df
dfoff = pd.read_csv('data/ccf_offline_stage1_train.csv')
dfon = pd.read_csv('data/ccf_online_stage1_train.csv')
dftest = pd.read_csv('data/ccf_offline_stage1_test_revised.csv')
 
dfoff.head(5)

print('有优惠卷，购买商品：%d' % dfoff[(dfoff['Date_received'] != 'null') & (dfoff['Date'] != 'null')].shape[0])
print('有优惠卷，未购商品：%d' % dfoff[(dfoff['Date_received'] != 'null') & (dfoff['Date'] == 'null')].shape[0])
print('无优惠卷，购买商品：%d' % dfoff[(dfoff['Date_received'] == 'null') & (dfoff['Date'] != 'null')].shape[0])
print('无优惠卷，未购商品：%d' % dfoff[(dfoff['Date_received'] == 'null') & (dfoff['Date'] == 'null')].shape[0])

dfoff.head(5)

print('Discount_rate 类型：\n',dfoff['Discount_rate'].unique())

dfoff = processData(dfoff)
dftest = processData(dftest)
# convert distance
dfoff['distance'] = dfoff['Distance'].replace('null', -1).astype(int)
print(dfoff['distance'].unique())
dftest['distance'] = dftest['Distance'].replace('null', -1).astype(int)
print(dftest['distance'].unique())

date_received = dfoff['Date_received'].unique()
date_received = sorted(date_received[date_received != 'null'])

date_buy = dfoff['Date'].unique()
date_buy = sorted(date_buy[date_buy != 'null'])

print('优惠卷收到日期从',date_received[0],'到',date_received[-1])
print('消费日期从',date_buy[0],'到',date_buy[-1])

def getWeekday(row):
    if row == 'null':
        return row
    else:
        return date(int(row[0:4]), int(row[4:6]), int(row[6:8])).weekday() + 1

dfoff['weekday'] = dfoff['Date_received'].astype(str).apply(getWeekday)
dftest['weekday'] = dftest['Date_received'].astype(str).apply(getWeekday)

# weekday_type :  周六和周日为1，其他为0
dfoff['weekday_type'] = dfoff['weekday'].apply(lambda x: 1 if x in [6,7] else 0)
dftest['weekday_type'] = dftest['weekday'].apply(lambda x: 1 if x in [6,7] else 0)

# change weekday to one-hot encoding 
weekdaycols = ['weekday_' + str(i) for i in range(1,8)]
#print(weekdaycols)

tmpdf = pd.get_dummies(dfoff['weekday'].replace('null', np.nan))
tmpdf.columns = weekdaycols
dfoff[weekdaycols] = tmpdf

tmpdf = pd.get_dummies(dftest['weekday'].replace('null', np.nan))
tmpdf.columns = weekdaycols
dftest[weekdaycols] = tmpdf


def label(row):
    if row['Date_received'] == 'null':
        return -1
    if row['Date'] != 'null':
        td = pd.to_datetime(row['Date'], format='%Y%m%d') - pd.to_datetime(row['Date_received'], format='%Y%m%d')
        if td <= pd.Timedelta(15, 'D'):
            return 1
    return 0

dfoff['label'] = dfoff.apply(label, axis=1)

df = dfoff[dfoff['label'] != -1].copy()
train = df[(df['Date_received'] < '20160516')].copy()
valid = df[(df['Date_received'] >= '20160516') & (df['Date_received'] <= '20160615')].copy()
print('Train Set: \n', train['label'].value_counts())
print('Valid Set: \n', valid['label'].value_counts())

# feature
original_feature = ['discount_rate','discount_type','discount_man', 'discount_jian','distance', 'weekday', 'weekday_type'] + weekdaycols
print('共有特征：',len(original_feature),'个')
print(original_feature)


def check_model(data, predictors):
    
    classifier = lambda: SGDClassifier(
        loss='log',  # loss function: logistic regression
        penalty='elasticnet', # L1 & L2
        fit_intercept=True,  # 是否存在截距，默认存在
        max_iter=100, 
        shuffle=True,  # Whether or not the training data should be shuffled after each epoch
        n_jobs=1, # The number of processors to use
        class_weight=None) # Weights associated with classes. If not given, all classes are supposed to have weight one.
 
    # 管道机制使得参数集在新数据集（比如测试集）上的重复使用，管道机制实现了对全部步骤的流式化封装和管理。
    model = Pipeline(steps=[
        ('ss', StandardScaler()), # transformer
        ('en', classifier())  # estimator
    ])
 
    parameters = {
        'en__alpha': [ 0.001, 0.01, 0.1],
        'en__l1_ratio': [ 0.001, 0.01, 0.1]
    }
 
    # StratifiedKFold用法类似Kfold，但是他是分层采样，确保训练集，测试集中各类别样本的比例与原始数据集中相同。
    folder = StratifiedKFold(n_splits=3, shuffle=True)
    
    # Exhaustive search over specified parameter values for an estimator.
    grid_search = GridSearchCV(
        model, 
        parameters, 
        cv=folder, 
        n_jobs=-1,  # -1 means using all processors
        verbose=1)
    grid_search = grid_search.fit(data[predictors], 
                                  data['label'])
    return grid_search




#def check_model(data, predictors):
#    
#    classifier = lambda: XGBClassifier(
#                learning_rate=0.1,
#                      n_estimators=500,         # 树的个数--1000棵树建立xgboost
#                      max_depth=5,               # 树的深度
#                      min_child_weight = 1,      # 叶子节点最小权重
#                      gamma=0.,                  # 惩罚项中叶子结点个数前的参数
#                      subsample=0.8,             # 随机选择80%样本建立决策树
#                      colsample_btree=0.8,       # 随机选择80%特征建立决策树
#                      )
#    # StratifiedKFold用法类似Kfold，但是他是分层采样，确保训练集，测试集中各类别样本的比例与原始数据集中相同。
#    model = Pipeline(steps=[
#        #('ss', StandardScaler()), # transformer
#        ('en', classifier())  # estimator
#    ])
#    
#    
#    parameters = {
#        'en__alpha': [1],
#        #'en__l1_ratio': [ 0.001, 0.01, 0.1]
#    }
#    folder = StratifiedKFold(n_splits=3, shuffle=True)    
#    # Exhaustive search over specified parameter values for an estimator.
#    grid_search = GridSearchCV(
#        model,  
#        parameters,
#        cv=folder, 
#        n_jobs=-1,  # -1 means using all processors
#        verbose=1)
#    grid_search = grid_search.fit(data[predictors], 
#                                  data['label'])
#    
#    return grid_search


def check_model2(data, predictors):
    
    #params = {'max_depth':2, 'eta':0.1, 'silent':0, 'objective':'binary:logistic' }
    bst =XGBClassifier(max_depth=2, learning_rate=0.1, silent=True, objective='binary:logistic')
    param_test = {
    'n_estimators':range(1,21,1)
    }
    clf = GridSearchCV(estimator=bst,param_grid=param_test,scoring='accuracy',cv=5)
    clf.fit(data[predictors],data['label'])

    # 管道机制使得参数集在新数据集（比如测试集）上的重复使用，管道机制实现了对全部步骤的流式化封装和管理。
    return clf





predictors = original_feature
model = check_model(train, predictors)
y_valid_pred = model.predict_proba(valid[predictors])
valid1 = valid.copy()
valid1['pred_prob'] = y_valid_pred[:, 1]
valid1.head(5)


vg = valid1.groupby(['Coupon_id'])
aucs = []
for i in vg:
    tmpdf = i[1] 
    if len(tmpdf['label'].unique()) != 2:
        continue
    fpr, tpr, thresholds = roc_curve(tmpdf['label'], tmpdf['pred_prob'], pos_label=1)
    aucs.append(auc(fpr, tpr))
print(np.average(aucs))


y_test_pred = model.predict_proba(dftest[predictors])
dftest1 = dftest[['User_id','Coupon_id','Date_received']].copy()
dftest1['Probability'] = y_test_pred[:,1]
dftest1.to_csv('submit1.csv', index=False, header=False)
dftest1.head(5)

if not os.path.isfile('1_model.pkl'):
    with open('1_model.pkl', 'wb') as f:
        pickle.dump(model, f)
else:
    with open('1_model.pkl', 'rb') as f:
        model = pickle.load(f)
