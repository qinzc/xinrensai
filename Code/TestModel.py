from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

import pandas as pd
import numpy as np
def TestModel(model):
    df = pd.read_csv('../Data/zhengqi_test.txt', sep='\t')
    y_pred = model.predict(df)
    print(y_pred)
    np.savetxt('output.txt',y_pred)


def nTestModel(model,test_x):
    y_pred = model.predict(test_x)
    print(y_pred)
    np.savetxt('output121212.txt',y_pred)



