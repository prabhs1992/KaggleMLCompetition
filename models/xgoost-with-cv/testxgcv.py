import numpy as np
import pandas as pd
import scipy as sc
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn import grid_search

data=pd.read_csv('../input/train.csv',sep=",")
data = data.replace(-999999,2)
numCol=len(data.columns)
numRow=len(data)
target=[]
target=data["TARGET"]
finData=[]
id=[]
id=data["ID"]
finData=data.iloc[:,1:numCol-1]
numCol=len(finData.columns)
numRow=len(finData)

data_test=pd.read_csv('../input/test.csv',sep=",")
data_test = data_test.replace(-999999,2)
id_test=[]
id_test=data_test["ID"]
numColt=len(data_test.columns)
numRowt=len(data_test)
finDatat=[]
finDatat=data_test.iloc[:,1:numColt]
numColt=len(finDatat.columns)
numRowt=len(finDatat)

scaler=preprocessing.StandardScaler().fit(finData)
train_scaled=scaler.transform(finData)
test_scaled=scaler.transform(finDatat)

p=PCA(n_components=train_scaled.shape[1])
p.fit(train_scaled)
trainX=p.transform(train_scaled)
testX=p.transform(test_scaled)

tr=trainX[:,0:142]
te=testX[:,0:142]

import xgboost as xgb

#crossvalidation steps might take a long time to run
clf = xgb.XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.85,
       gamma=0, learning_rate=0.02, max_delta_step=0, max_depth=6,
       min_child_weight=4, missing=None, n_estimators=567, nthread=4,
       objective='binary:logistic', reg_alpha=0, reg_lambda=1,
       scale_pos_weight=1, seed=4242, silent=True, subsample=0.95)

params = clf.get_xgb_params()
#create a Dmatrix required for the cv function, also we are doing 5-fold cv as that is the most popular
#we tried with metrics = auc and logloss , while logloss gives n_estimator of 440 auc gives 441
result = xgb.cv(params,xgb.DMatrix(tr,target),num_boost_round=params['n_estimators'],nfold=5,metrics=['auc'],early_stopping_rounds=50)
#get n_estimator from cv result and set it
clf.set_params(n_estimators = result.shape[0])

# tuning max_depth and min_child_weight parameters using gridsearch
test_grid = {'max_depth':range(3,11,1),'min_child_weight':range(1,7,1)}
#perform grid search on the 63 possible combinations and try to get the best out of it
gs = grid_search.GridSearchCV(estimator=clf,param_grid=test_grid,scoring='roc_auc',n_jobs=4,iid=False,cv=5)
gs.fit(tr,target)

#update clf with the new best parameter values and then fit the training data before classifying
clf.set_params(max_depth = gs.best_params_['max_depth'],min_child_weight = gs.best_params_['min_child_weight'])

clf.fit(tr, target, eval_metric="auc", eval_set=[(tr, target)])

y_pred2 = clf.predict_proba(te)

#finally save results
submission = pd.DataFrame({"ID":id_test, "TARGET":y_pred2[:,1]})
submission.to_csv("submission.csv", index=False)