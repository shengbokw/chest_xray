import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB

# read all the dataset
# read validation set
x1 = pd.read_csv("../normal_val.csv", sep=',', header=None)
y1 = [0] * len(x1.index)
x2 = pd.read_csv("../PNEUMONIA_val.csv", sep=',', header=None)
y2 = [1] * len(x2.index)
X_val = x1.append(x2)
Y_val = np.array(y1 + y2)

# read training set
x3 = pd.read_csv("../normal_train.csv", sep=',', header=None)
x3['label'] = 0
x4 = pd.read_csv("../PNEUMONIA_train.csv", sep=',', header=None)
x4['label'] = 1
X_train = x3.append(x4)

# drop the row if they have zero value for lung size
X_train = X_train.drop(X_train[X_train[260]==0].index)
# print(X_train[X_train[260]==0])

X_train = X_train.fillna(X_train.mean())
Y_train = X_train["label"]
# print(X_train.any(X_train.isnan(X_train)))
# print(np.all(np.isfinite(X_train)))

# read testing set
x5 = pd.read_csv("../normal_test.csv", sep=',', header=None)
x5['label'] = 0
x6 = pd.read_csv("../PNEUMONIA_test.csv", sep=',', header=None)
x6['label'] = 1
X_test = x5.append(x6)
X_test = X_test.drop(X_test[X_test[260] == 0].index)
X_test = X_test.fillna(X_test.mean())
Y_test = X_test["label"]



'''
gnb = GaussianNB()
y_pred_train = gnb.fit(X_train.iloc[:, :-5], Y_train).predict(X_train.iloc[:, :-5])
y_pred_test = gnb.fit(X_train.iloc[:, :-5], Y_train).predict(X_test.iloc[:, :-5])

# check the result
data = {'pred': y_pred_test, 'label': Y_test}
data2 = {'pred': y_pred_train, 'label': Y_train}
df = pd.DataFrame(data=data)
df2 = pd.DataFrame(data=data2)

# Y_train["predicted_train"] = y_pred_train
# Y_train["predicted_test"] = y_pred_test

print(df2)
'''
# evaluation, for fpr, tpr

# feature extraction

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train.iloc[:, :-1], Y_train)
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
raw_result = clf.predict(X_test.iloc[:, :-1])
raw_score = clf.score(X_test.iloc[:, :-1], Y_test)

# the problem of unbalance
# the problem of non-normal distribution ( the different weights problem)
# non-equal weights should be applied using other methods



