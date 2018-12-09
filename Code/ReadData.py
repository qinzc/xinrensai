import pandas as pd
import numpy as np



def ReadTrain():
    df=pd.read_csv('../Data/zhengqi_train.txt',sep='\t')
    x,y=df.iloc[:,0:38],df.iloc[:,[38]]
    return x,y

def ReadAll():
    train_df = pd.read_csv('../Data/zhengqi_train.txt', sep='\t')
    train_x, train_y = train_df.iloc[:, 0:38], train_df.iloc[:, [38]]
    test_x=pd.read_csv('../Data/zhengqi_test.txt', sep='\t')
    x = train_x.append(test_x)

    ser = train_df.iloc[:, 0:38].corrwith(train_df.target)
    dropCols = []
    for i in ser.index:
        print ser[i]
        if abs(ser[i]) < 0.0:
            dropCols.append(i)
    print(dropCols)
    x = x.drop(columns=dropCols)

    bp = x.boxplot(return_type='dict')
    t = [whk.get_ydata() for whk in bp["whiskers"]]
    segs = []
    for i in range(len(t) / 2):
        segs.append([t[i * 2][1], t[i * 2 + 1][1]])

    x.index = range(len(x))

    def myFloor(t, segs, j):
        j = int(j)
        #    print(t,j,segs[j][0],segs[j][1])
        if t < segs[j][0]:
            return segs[j][0]
        elif t > segs[j][1]:
            return segs[j][1]
        else:
            return t

    for j, col in enumerate(x.columns):
        print j, col
        for i in x.index:
            x.iloc[i, j] = myFloor(x.iloc[i, j], segs, j)

    #x=(x-x.mean())/(x.max()-x.min())
    x = (x - x.min()) / (x.max() - x.min())
    ntrain_x = x.iloc[:2888, :]
    ntest_x = x.iloc[2888:, :]
    ntrain_y=train_y
    return ntrain_x,ntest_x,ntrain_y

if __name__=='__main__':
    ReadData()