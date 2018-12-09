
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm
from sklearn.ensemble import GradientBoostingRegressor as GBR

def TrainModel(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    linreg = LinearRegression()
    linreg.fit(x_train, y_train)
    y_pred=linreg.predict(x_test)
    print "MSE:", metrics.mean_squared_error(y_test, y_pred)
    return linreg

def nTrainModel(train_x,train_y):
    #clf = svm.SVR(kernel='rbf', gamma=0.04, C=100)
    clf = LinearRegression()
    clf.fit(train_x, train_y)
    return clf


def TrainModel_SVR(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    clf = svm.SVR(kernel='rbf', gamma=0.04, C=100)
    clf.fit(x_train, y_train)
    y_pred=clf.predict(x_test)
    print "MSE:", metrics.mean_squared_error(y_test, y_pred)
    return clf

def TrainModel_GBR(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    clf = GBR()
    clf.fit(x_train, y_train)
    y_pred=clf.predict(x_test)
    print "MSE:", metrics.mean_squared_error(y_test, y_pred)
    return clf