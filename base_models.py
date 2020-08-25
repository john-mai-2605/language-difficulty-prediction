import pandas as pd
import re
import numpy as np
import cmudict
from sklearn import *
import matplotlib.pyplot as plt
from collections import Counter
import pickle

from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import RobustScaler, OrdinalEncoder, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import GridSearchCV

from sklearn.cluster import KMeans
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.linear_model import LinearRegression, SGDRegressor, PassiveAggressiveRegressor, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.kernel_ridge import KernelRidge

import xgboost as xgb 
import lightgbm as lgb

print("Load data...")
with open('data.pkl', 'rb') as f:
    data = pickle.load(f)
X_train, y_train_clf, y_train = data['train'].values()
X_test, y_test_clf, y_test = data['test'].values()
class Regressor:
    def __init__(self, reg):
        self.reg = reg
    def fit(self, X, y, **kwargs):
        return self.reg.fit(X, y, **kwargs)
    def predict(self, X_test):
        return self.reg.predict(X_test)
    def evaluate(self, y_test, y_pred): 
        mse = metrics.mean_squared_error(y_test, y_pred)
        r2 = metrics.r2_score(y_test, y_pred)
        print(np.sqrt(mse), r2)
        clf_test = [map_to_class(y) for y in y_test]
        clf_pred = [map_to_class(y) for y in y_pred]
        print(metrics.confusion_matrix(clf_test, clf_pred))
        print(metrics.classification_report(clf_test, clf_pred))
    def tune(self, X_train, Y_train, param_grid,
             n_folds = 10, result_filename = 'reg_tuning_results.csv'):
        grid_cv = GridSearchCV(estimator = self.reg,
                           param_grid = param_grid,
                           cv = n_folds,
                           scoring = 'neg_mean_squared_error',
                           verbose = 0,
                           n_jobs = -1
                          )
        grid_cv.fit(X_train, Y_train)
        # save results
        tuning_results = pd.DataFrame(grid_cv.cv_results_)
        tuning_results.to_csv(result_filename)

        # set best params
        best_params = grid_cv.best_params_
        self.reg.set_params(**best_params)

    def run(self, X, y, X_test, y_test, **kwargs):
        fitted_model = self.fit(X, y, **kwargs)
        y_pred = self.predict(X_test)
        self.evaluate(y_test, y_pred)
        try:
            importance = list(zip(fitted_model.feature_importances_, feature_columns))
            importance.sort(reverse=True)
            print(importance)
        except:
            pass
        return y_pred
    def select(self, X_train, X_test, k = 50):
        self.selector = SelectFromModel(self.reg, prefit = True, threshold = -np.inf, max_features = k)
        return self.selector.transform(X_train), self.selector.transform(X_test)

def map_to_class(score, s1 = 2/3, s2 = 1/3):
    if score > s1:
        return 2
    if score > s2:
        return 1
    return 0

if __name__ == '__main__':
    print("---------Regressor---------")
    models = [ElasticNet(), Lasso(), 
            GradientBoostingRegressor(), BayesianRidge(), LassoLarsIC(), 
            RandomForestRegressor(), xgb.XGBRegressor(), lgb.LGBMRegressor(),
            svm.SVR(), neural_network.MLPRegressor(), LinearRegression(), 
            SGDRegressor(), PassiveAggressiveRegressor(), HuberRegressor()]
    EN_param_grid = {'alpha': [0.001, 0.01, 0.0001], 'copy_X': [True], 'l1_ratio': [0.6, 0.7], 'fit_intercept': [True], 'normalize': [False], 
                             'precompute': [False], 'max_iter': [300, 3000], 'tol': [0.001], 'selection': ['random', 'cyclic'], 'random_state': [None]}
    LASS_param_grid = {'alpha': [0.001, 0.0001, 0.00001, 0.000001], 'copy_X': [True], 'fit_intercept': [True, False], 'normalize': [False], 'precompute': [False], 
                        'max_iter': [300, 1000, 3000], 'tol': [0.1, 0.01, 0.001], 'selection': ['random'], 'random_state': [42]}
    GB_param_grid = {'loss': ['huber'], 'learning_rate': [0.1, 0.01, 0.001], 'n_estimators': [3000], 'max_depth': [3, 10], 
                                            'min_samples_split': [0.0025], 'min_samples_leaf': [5]}
    BR_param_grid = {'n_iter': [200], 'tol': [0.00001], 'alpha_1': [0.00000001], 'alpha_2': [0.000005], 'lambda_1': [0.000005], 
                     'lambda_2': [0.00000001], 'copy_X': [True]}
    LL_param_grid = {'criterion': ['aic'], 'normalize': [True], 'max_iter': [100, 1000], 'copy_X': [True], 'precompute': ['auto'], 'eps': [0.000001, 0.00001, 0.0001]}
    RFR_param_grid = {'n_estimators': [50, 500], 'max_features': ['auto'], 'max_depth': [None], 'min_samples_split': [5], 'min_samples_leaf': [2]}
    XGB_param_grid = {'max_depth': [3, 10], 'learning_rate': [0.1, 0.05, 0.5], 'n_estimators': [000], 'booster': ['gbtree'], 'gamma': [0], 'reg_alpha': [0.1, 0.01],
                      'reg_lambda': [0.7], 'max_delta_step': [0], 'min_child_weight': [1], 'colsample_bytree': [0.5], 'colsample_bylevel': [0.2],
                      'scale_pos_weight': [1]}
    LGB_param_grid = {'objective': ['regression'], 'learning_rate': [0.05, 0.1, 0.5], 'n_estimators': [300, 3000]}
    SVR_param_grid = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
    MLP_param_grid = {'hidden_layer_sizes': [(100,), (100, 10)], 'random_state': [42], 'max_iter': [100, 1000],  'alpha': [0.01, 0.001, 0.0001]}
    LR_param_grid = {}
    GDR_param_grid = {
                    'loss': ['squared_loss', 'huber', 'squared_epsilon_insensitive', 'epsilon_insensitive'],
                    'penalty': ['l2', 'elasticnet', 'l1'],
                    'l1_ratio': [0.7, 0.8, 0.2, 0.5, 0.03],
                    'learning_rate': ['invscaling', 'constant', 'optimal'],
                    'alpha': [1e-01, 1e-2, 1e-03, 1e-4, 1e-05],
                    'epsilon': [1e-01, 1e-2, 1e-03, 1e-4, 1e-05],
                    'tol': [0.001, 0.003],
                    'eta0': [0.01, 1e-1, 1e-03, 1e-4, 1e-05],
                    'power_t': [0.5]}
    PAR_param_grid = {'loss': ['squared_epsilon_insensitive', 'epsilon_insensitive'], 
                      'C': [0.001, 0.005, 0.003], 'max_iter': [1000], 'epsilon': [0.00001, 0.00005],
                      'tol': [1e-03, 1e-05,1e-02, 1e-01, 1e-04, 1e-06]}
    HR_param_grid = {'max_iter': [2000], 'alpha': [0.0001, 5e-05, 0.01, 0.00005, 0.0005, 0.5, 0.001], 
                  'epsilon': [1.005, 1.05, 1.01, 1.001], 'tol': [1e-01, 1e-02]}
    params_grids = [EN_param_grid, LASS_param_grid, GB_param_grid, BR_param_grid, 
                    LL_param_grid, RFR_param_grid, XGB_param_grid, LGB_param_grid, 
                    SVR_param_grid, MLP_param_grid, LR_param_grid, GDR_param_grid,
                    PAR_param_grid, HR_param_grid]
    regs = []
    params = []
    for model, param_grid in zip(models, params_grids):
        print(model.__class__.__name__)
        regressor = Regressor(model)
        print('Start tuning')
        regressor.tune(X_train, y_train, param_grid, result_filename = model.__class__.__name__ + ' hyperparameters.csv')
        print('Finish tuning')
        params.append(regressor.reg.get_params())
        y = regressor.run(X_train, y_train, X_test, y_test)
        regs.append(regressor)

    with open('regs.pkl', 'wb') as f:
        pickle.dump(regs, f)
    with open('params.pkl', 'wb') as f:
        pickle.dump(params, f)