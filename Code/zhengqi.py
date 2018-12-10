
# coding: utf-8

import pandas as pd
import numpy as np
df=pd.read_csv('../Data/zhengqi_train.txt',sep='\t')



#我们发现有很多异常点
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(20.0, 10.0))
#g = sns.boxplot(data=df)
#plt.show()
bp=df.boxplot(return_type='dict')
t=[whk.get_ydata() for whk in bp["whiskers"]]
# plt.savefig('./sss.png')
print("delete")
print(123)
# In[ ]:

print('123')



