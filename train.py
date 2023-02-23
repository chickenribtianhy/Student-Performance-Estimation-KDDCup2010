import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyRegressor
import lightgbm
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV


train = pd.read_csv('./data/preprossed_train.csv', sep='\t')
test = pd.read_csv('./data/preprossed_test.csv', sep='\t')


# ======================================data generation====================================== #

train_x = train.dropna()
train_y = np.array(train_x['Correct First Attempt']).astype(int)
train_x = train_x.drop(['Correct First Attempt'],axis = 1)

test_x = test.dropna()
test_y = np.array(test_x['Correct First Attempt']).astype(int)
test_x = test_x.drop(['Correct First Attempt'],axis = 1)


# ======================================training====================================== #


def normalize(x):
    return x / np.linalg.norm(x, ord=2, axis=1, keepdims=True)


# RMSE error function
def RMSE(x, y):
    return np.sqrt(np.mean(np.square(np.subtract(x, y))))


def train_models():
    global tree
    # MLPRegressor
    model = MLPRegressor(hidden_layer_sizes=(100, 5, 100), activation='tanh', solver='adam')
    model.fit(train_x, train_y)
    test_res = model.predict(test_x)

    print("MLPRegressor : %f" % RMSE(test_res, test_y))

    # decision tree
    model = tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)
    test_res = model.predict(test_x)

    print("decision tree : %f" % RMSE(test_res, test_y))

    # RandomForestRegressor

    model = RandomForestRegressor()
    model.fit(train_x, train_y)
    test_res = model.predict(test_x)

    print("RandomForest : %f" % RMSE(test_res, test_y))
    # n_estimator=190 => 3.53911, 0.354522, 0.354397
    # n_estimator=170 => 3.54011, 3.54304, 0.354121
    # n_estimator=205 => 0.354135, 0.354228, 0.354194
    # n_estimator=215 => 0.355086

    model = RandomForestRegressor(n_estimators=170, max_depth=15, max_leaf_nodes=900)
    model.fit(train_x, train_y)
    test_res = model.predict(test_x)

    print("RandomForest opt : %f" % RMSE(test_res, test_y))

    # RandomForestRegressor optimize

    # n_estimators = range(160, 210, 5)
    # n_estimators = range(170, 220, 5)
    # random_forest_para = {'n_estimators':n_estimators}
    # model = GridSearchCV(RandomForestRegressor(), random_forest_para, n_jobs=-1)
    # model.fit(train_x, train_y)
    # test_res = model.predict(test_x)

    # print("optimze RandomForest : %f" % RMSE(test_res, test_y))
    # print("best parameters", model.best_params_ )
    # Adaboost

    model = AdaBoostRegressor()
    model.fit(train_x, train_y)
    test_res = model.predict(test_x)

    print("Adaboost : %f" % RMSE(test_res, test_y))

    # XGBoost

    model = XGBClassifier()
    model.fit(train_x, train_y)
    test_res = model.predict(test_x)

    print("XGBoost : %f" % RMSE(test_res, test_y))

    # lightgbm

    model = lightgbm.LGBMClassifier()
    model.fit(train_x, train_y)
    test_res = model.predict(test_x)

    print("lightgbm : %f" % RMSE(test_res, test_y))

    # Gradient Decision Tree

    model = GradientBoostingClassifier(n_estimators=200)
    model.fit(train_x, train_y)
    test_res = model.predict(test_x)

    print("Gradient Decision Tree : %f" % RMSE(test_res, test_y))

    # Logistic Regression

    model = LogisticRegression(penalty='l2')
    model.fit(train_x, train_y)
    test_res = model.predict(test_x)

    print("Logistic Regression : %f" % RMSE(test_res, test_y))

    # DummyRegressor()

    model = DummyRegressor()
    model.fit(train_x, train_y)
    test_res = model.predict(test_x)

    print("Dummy Regression : %f" % RMSE(test_res, test_y))

    # KNN

    model = neighbors.KNeighborsRegressor()
    model.fit(train_x, train_y)
    test_res = model.predict(test_x)

    print("KNN : %f" % RMSE(test_res, test_y))


    tree = tree.DecisionTreeClassifier()
    model = BaggingRegressor(base_estimator=tree, n_estimators=100, max_samples=1.0, bootstrap=True)
    model.fit(train_x, train_y)
    test_res = model.predict(test_x)

    print("tree bagging : %f" % RMSE(test_res, test_y))



    knn = neighbors.KNeighborsRegressor()
    model = BaggingRegressor(base_estimator=knn, n_estimators=100, max_samples=1.0, bootstrap=True)
    model.fit(train_x, train_y)
    test_res = model.predict(test_x)

    print("knn bagging : %f" % RMSE(test_res, test_y))

    Dummy = DummyRegressor()
    model = BaggingRegressor(base_estimator=Dummy, n_estimators=100, max_samples=1.0, bootstrap=True)
    model.fit(train_x, train_y)
    test_res = model.predict(test_x)

    print("Dummy bagging : %f" % RMSE(test_res, test_y))

    # ======================================export csv====================================== #

train_models()

def export():
    test_x = test
    test_y = np.array(test_x['Correct First Attempt']).astype(float)
    test_x = test_x.drop(['Correct First Attempt'], axis=1)
    
    model = RandomForestRegressor(n_estimators=170, max_depth=15, max_leaf_nodes=900)
    model.fit(train_x, train_y)
    test_res = model.predict(test_x)
    # print("optimze RandomForest : %f" % RMSE(test_res, test_y))
    for i, val in enumerate(test_y):
        if np.isnan(val):
            test_y[i] = test_res[i]
    new_test = pd.read_csv('./data/test.csv', sep='\t')
    new_test['Correct First Attempt'] = test_y
    new_test.to_csv('output.csv', sep='\t', index=False)

export()