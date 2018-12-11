from ReadData import ReadAll
from TrainModel import nTrainModel
from TrainModel import TrainModel
from TrainModel import *
from TestModel import nTestModel

# x,y=ReadData()
# model=TrainModel(x,y)
# TestModel(model)


train_x,test_x,train_y=ReadAll()
model=nTrainModel_Poly(train_x,train_y)
#model=TrainModel_GBR_Liner(train_x,train_y)
#model = nTrainModel(train_x,train_y)
#model=TrainModel_SVR(train_x,train_y)
#model=TrainModel_GBR(train_x,train_y)
nTestModel(model,test_x)