import os
import numpy as np
from sklearn import svm
from numpy import genfromtxt
from sklearn.svm import SVC
from collections import Counter

"""
File name: svm_script_new_largedb.py
    Author: Isabella Douzoglou
    Date created: 5/03/2019
    Date last modified: 28/03/2019
    Python Version: 2.7
"""

def create_validation_array(X1_size, X2_size):
    # dynamic y creation based on the size of X1 and X2
    y_P = [1.0 for x in range(0, X1_size[0])]
    y_N = [0.0 for y in range(0, X2_size[0])]

    # concatenate the two
    y = y_P + y_N

    # print("condadenated y" + str(y))
    print('y_P length ' + str(len(y_P)))
    print("y_N length " + str(len(y_N)))

    return y

path1 = "pneumonia_train_small.csv"
path2 = "normal_train_small.csv"
#X1 = pd.read_csv(path1, header=None, dtype=np.float64, delimiter=',')

X1 = genfromtxt(path1, delimiter=',')
X2 = genfromtxt(path2, delimiter=',')

X1_size = X1.shape
X2_size = X2.shape
print("size x1" + str(X1_size))
print("size x2" + str(X2_size))

#create array of the right size
y = create_validation_array(X1_size, X2_size)

# combine x 1 and x 2
X = np.concatenate((X1, X2))

# make NaN to 0
X = np.nan_to_num(X)

# creates an svn model
clf = svm.SVC(kernel="linear")
clf.fit(X, y)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# open test data
path1 = "pneumonia_test_small.csv"
path2 = "normal_test_small.csv"
test_PNE = genfromtxt(path1, delimiter=',')
test_NOR = genfromtxt(path2, delimiter=',')
print("test_PNE"+ str(test_PNE.shape))
print("test_NOR"+ str(test_NOR.shape))


# concatenate them
dataNan = np.concatenate((test_PNE, test_NOR))
print("size" + str(dataNan.shape))

# convert nan to num
data = np.nan_to_num(dataNan)


# ask model to predict data, returns an array
answer = clf.predict(data)
# print("answer " + str(answer) + " answer size = " + str(len(answer)))

count_pneumonia_correct = float('-inf')
count_normal_correct = float('-inf')

PNE = answer[0:391]
NOR = answer[391:626]

countsPNE = Counter(PNE)
countsNOR = Counter(NOR)
count_pneumonia_correct = countsPNE[1.0]
count_normal_correct = countsNOR[0.0]

percentage_PNE = float(count_pneumonia_correct)/391 * 100
percentage_NOR = float(count_normal_correct)/235 * 100

print("% pneumonia " + str(percentage_PNE))
print("% normal " + str(percentage_NOR))

