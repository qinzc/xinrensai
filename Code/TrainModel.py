
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import ElasticNet

def TrainModel(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    linreg = LinearRegression()
    #linreg=ElasticNet(random_state=0)
    linreg.fit(x_train, y_train)
    y_pred=linreg.predict(x_test)
    print("MSE:", metrics.mean_squared_error(y_test, y_pred))
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
    print("MSE:", metrics.mean_squared_error(y_test, y_pred))
    return clf

def TrainModel_GBR(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    #clf = GBR()
    clf=GBR(alpha=0.9, criterion='friedman_mse', init=None, \
                                  learning_rate=0.03, loss='huber', max_depth=15,\
                                  max_features='sqrt', max_leaf_nodes=None,\
                                  min_impurity_decrease=0.0, min_impurity_split=None, \
                                  min_samples_leaf=10, min_samples_split=40,\
                                  min_weight_fraction_leaf=0.0, n_estimators=300, \
                                  presort='auto', random_state=10, subsample=0.8, verbose=0, \
                                  warm_start=False)
    clf.fit(x_train, y_train)
    y_pred=clf.predict(x_test)
    print("MSE:", metrics.mean_squared_error(y_test, y_pred))
    return clf

def TrainModel_Poly(x,y):
    polynomial = PolynomialFeatures(degree=4)  # 二次多项式
    x_transformed = polynomial.fit_transform(x)  # x每个数据对应的多项式系数
    x_train, x_test, y_train, y_test = train_test_split(x_transformed, y, test_size=0.3, random_state=0)
    #linreg = LinearRegression()
    linreg = ElasticNet(random_state=0,alpha=0.0001,l1_ratio = 0.5)
    linreg.fit(x_train, y_train)
    print(linreg.coef_,linreg.intercept_)
    y_pred=linreg.predict(x_test)
    print("MSE:", metrics.mean_squared_error(y_test, y_pred))
    return linreg

def nTrainModel_Poly(x,y):
    polynomial = PolynomialFeatures(degree=4)  # 二次多项式
    x_transformed = polynomial.fit_transform(x)  # x每个数据对应的多项式系数
    #x_train, x_test, y_train, y_test = train_test_split(x_transformed, y, test_size=0.3, random_state=0)
    #linreg = LinearRegression()
    linreg = ElasticNet(random_state=0,alpha=0.0001,l1_ratio = 0.5)
    linreg.fit(x, y)
    print(linreg.coef_,linreg.intercept_)
    #y_pred=linreg.predict(x_test)
    #print("MSE:", metrics.mean_squared_error(y_test, y_pred))
    return linreg


def TrainModel_GBR_Liner(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    #clf = GBR()
    clf=GBR(alpha=0.9, criterion='friedman_mse', init=None, \
                                  learning_rate=0.03, loss='huber', max_depth=15,\
                                  max_features='sqrt', max_leaf_nodes=None,\
                                  min_impurity_decrease=0.0, min_impurity_split=None, \
                                  min_samples_leaf=10, min_samples_split=40,\
                                  min_weight_fraction_leaf=0.0, n_estimators=300, \
                                  presort='auto', random_state=10, subsample=0.8, verbose=0, \
                                  warm_start=False)
    clf.fit(x_train, y_train)
    y_pred1=clf.predict(x_test)

    linreg = LinearRegression()
    linreg.fit(x_train, y_train)
    y_pred2 = linreg.predict(x_test)

    y_pred=0.5*y_pred1+0.5*y_pred2.T
    print("MSE:", metrics.mean_squared_error(y_test, y_pred[0]))
    return clf