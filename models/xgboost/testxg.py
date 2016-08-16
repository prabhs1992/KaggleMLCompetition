import numpy as np
import pandas as pd
import scipy as sc
from sklearn import preprocessing
from sklearn.decomposition import PCA

# the script can also be seen posted on kaggle (where we ran the script directly).
# at https://www.kaggle.com/prabhanjan1992/santander-customer-satisfaction/46run/run/224575/code

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

#n_estimators = 441 was obained by Cross validation step 1 mentioned in the report.
clf = xgb.XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.85,
       gamma=0, learning_rate=0.02, max_delta_step=0, max_depth=6,
       min_child_weight=4, missing=None, n_estimators=441, nthread=4,
       objective='binary:logistic', reg_alpha=0, reg_lambda=1,
       scale_pos_weight=1, seed=4242, silent=True, subsample=0.95)

clf.fit(tr, target, eval_metric="auc", eval_set=[(tr, target)])

y_pred2 = clf.predict_proba(te)

submission = pd.DataFrame({"ID":id_test, "TARGET":y_pred2[:,1]})
submission.to_csv("submission.csv", index=False)