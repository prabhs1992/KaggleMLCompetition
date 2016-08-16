import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.metrics import roc_auc_score

data=pd.read_csv('/home/prabhanjan/Downloads/te/train.csv',sep=",")
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

data_test=pd.read_csv('/home/prabhanjan/Downloads/te/test.csv',sep=",")
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

X_train, X_test, y_train, y_test = cross_validation.train_test_split(tr, target, random_state=1301, stratify=target, test_size=0.35)
#max_features should be square root of features = 12
rf = RandomForestClassifier(n_estimators=470, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=12, max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)

#validation test
rf.fit(X_train, y_train)
print('Overall AUC:', roc_auc_score(target, rf.predict_proba(tr)[:,1]))
y_pred = rf.predict_proba(X_test)
max=np.argmax(y_pred,axis=1)
lab = y_test.as_matrix()
correct_pred = 0
for i in range(len(max)):
    if max[i] == lab[i]:
        correct_pred += 1
print((100*correct_pred) / len(lab))

#training on entire data
rf.fit(tr,target)

#testing
y_pred2 = rf.predict_proba(te)
submission = pd.DataFrame({"ID":id_test, "TARGET":y_pred2[:,1]})
submission.to_csv("submission.csv", index=False)