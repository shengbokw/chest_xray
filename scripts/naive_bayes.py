import pandas as pd

x1 = pd.read_csv("../normal_val.csv", sep=',', header=None)
#x1 = x1.to_numpy()
x2 = pd.read_csv("../PNEUMONIA_val.csv", sep=',', index_col=None)
# x2 = x2.to_numpy()
# X = x1 + x2
print(x1)


