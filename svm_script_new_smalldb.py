import os
import numpy as np
from sklearn import svm
from numpy import genfromtxt
from sklearn.svm import SVC


"""
def create_validation_array(X1_size, X2_size):
    # dynamic y creation based on the size of X1 and X2
    y_P = [1.0 for x in range(0, X1_size[0])]
    y_N = [0.0 for y in range(0, X2_size[0])]
    # concatenate the two
    y = y_P + y_N
    print("condadenated y" + str(y))
    return y
"""

path1 = "pneumonia_val_small.csv"
path2 = "normal_val_small.csv"
#X1 = pd.read_csv(path1, header=None, dtype=np.float64, delimiter=',')
#X2 = pd.read_csv(path2, header=None, dtype=np.float64, delimiter=',')

# 8 x 261 becomes 16 x 261
X1 = genfromtxt(path1, delimiter=',')
X2 = genfromtxt(path2, delimiter=',')

'''
X1_size = X1.shape
X2_size = X2.shape
print("type "+ str(type(X1)))
print(X1.shape)
print(X2.shape)
print(X1)
print(X2)
print("len x1: " + str(X1_size[0]))
print("len x2: " + str(X2_size[0]))
'''


#create data for testing
#train with five tes with 3
train_data_PNE = X1[0:4]
train_data_NOR = X2[0:4]
test_data_PNE = X1[5:7]
test_data_NOR = X2[5:7]
# print("train data NOR" + str(train_data_NOR))
# print("test data NOR" + str(test_data_NOR))


X1_size = train_data_NOR.shape
X2_size = train_data_PNE.shape
print("size x1" + str(X1_size))
print("size x2" + str(X2_size))


#create array of the right size
y = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]

# combine x 1 and x 2
X = np.concatenate((train_data_PNE, train_data_NOR))
# print(X.shape)
# print(X)

# make NaN to 0
# X = np.nan_to_num(X)
# print(X)

# creates an svn model
clf = svm.SVC(kernel="linear")
clf.fit(X, y)
"""
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
"""
print("model created")

data = np.concatenate((test_data_PNE, test_data_NOR))
print("size" + str(data.shape))

# ask model to predict data, returns an array
answer = clf.predict(data)
print("answer" + str(answer))