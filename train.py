import pandas as pd
import re
import numpy as np
import cmudict
from sklearn import *
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

unigrams_df = pd.read_csv('unigram_freq.csv', index_col = 'word')

class Dataset:
	def __init__(self, path):
		self.data = pd.read_csv(path)
		self.X = self.data[[
			'length', 
			'aveLength', 'maxLength', 'minLength',
			'aveFreq', 'maxFreq', 'minFreq', 
			'aveDepth', 'maxDepth', 'minDepth', 
			'aveDensity', 'minDensity', 'maxDensity',
			'aveAmbiguity', 'minAmbiguity', 'maxAmbiguity', 
			'wpm', 'elapse_time', 'accentTag', 'genreTag']]
		self.y = self.data['error_rate']
		self.scaler = RobustScaler()
		self.imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
		self.enc_oh = OneHotEncoder()
		self.enc_ord = OrdinalEncoder()
	def split(self, test_ratio):
		return model_selection.train_test_split(self.X, self.y, test_size = test_ratio, random_state=42)
	
	def encode(self):
		self.X['accentTag'] = self.enc_ord.fit_transform(self.X['accentTag'].to_numpy().reshape(-1, 1))
		self.X['genreTag'] = self.enc_ord.fit_transform(self.X['genreTag'].to_numpy().reshape(-1, 1))
		
	def normalize(self, X_train, X_test):
		self.imputer.fit(X_train)
		X_train = self.imputer.transform(X_train)
		X_test = self.imputer.transform(X_test)
		self.scaler.fit(X_train) 
		return self.scaler.transform(X_train), self.scaler.transform(X_test)

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

class Classifier:
	def __init__(self, clf):
		self.clf = clf
	def fit(self, X, y, **kwargs):
		return self.clf.fit(X, y, **kwargs)
	def predict(self, X_test):
		return self.clf.predict(X_test)		
	def evaluate(self, y_test, y_pred):	
		print(metrics.confusion_matrix(y_test, y_pred))
		print(metrics.classification_report(y_test, y_pred))

def map_to_class(score, s1 = 2/3, s2 = 1/3):
	if score > s1:
		return 2
	if score > s2:
		return 1
	return 0

data = Dataset('../processed_data.csv')
data.encode()
X_train, X_test, y_train, y_test = data.split(0.2)
X_train, X_test = data.normalize(X_train, X_test)


print("---------Regressor---------")

# Decision tree
print("Decision tree")
DT = Regressor(tree.DecisionTreeRegressor(max_depth = 3))
DT.fit(X_train, y_train)
y_pred = DT.predict(X_test)
DT.evaluate(y_test, y_pred)
# Boosted tree
import xgboost as xgb 
param = {'max_depth': 5, 
	'eta': 0.3, 
	'objective': 'reg:squarederror', 
	'verbosity' : 0}


# Not using wrapper:
print("XGB Tree")
dtrain = xgb.DMatrix(X_train, label = y_train)
dtest = xgb.DMatrix(X_test, label = y_test)
BT = Regressor(xgb.train(param, dtrain, 1000, 
	[(dtrain, 'train'), (dtest, 'eval')],
	early_stopping_rounds = 10))
y_pred = BT.reg.predict(dtest)
xgb.plot_importance(BT.reg)
BT.evaluate(y_test, y_pred)
plt.show()

# Using sklearn wrapper
# print("XGB Tree")
# BT = Regressor(xgb.XGBModel(**param))
# BT.fit(X_train, y_train, **{'eval_set':[(X_train, y_train), (X_test, y_test)],
#         'early_stopping_rounds': 10,
#         'verbose' : False})
# y_pred = BT.predict(X_test)
# BT.evaluate(y_test, y_pred)
# xgb.plot_importance(BT.reg)
# plt.show()

# RF
print("Random Forest")
RF = Regressor(ensemble.RandomForestRegressor(max_depth=4))
RF.fit(X_train, y_train)
y_pred = RF.predict(X_test)
RF.evaluate(y_test, y_pred)

# SVM
print("Support vector machine")
SVM = Regressor(svm.SVR())
SVM.fit(X_train, y_train)
y_pred = SVM.predict(X_test)
SVM.evaluate(y_test, y_pred)

# MLP
print("Multi Perceptron Layer (NN)")
NN = Regressor(neural_network.MLPRegressor(hidden_layer_sizes = (100,), random_state=1, max_iter=100, alpha = 0.001))
NN.fit(X_train, y_train)
y_pred = NN.predict(X_test)
NN.evaluate(y_test, y_pred)

print("--------Classifier----------")
y_train_clf = [map_to_class(y) for y in y_train]
y_test_clf = [map_to_class(y) for y in y_test]
# Decision tree
DT = Classifier(tree.DecisionTreeClassifier(max_depth = 3))
DT.fit(X_train, y_train_clf)
y_pred = DT.predict(X_test)
DT.evaluate(y_test_clf, y_pred)
# Boosted tree
param = {'max_depth': 5, 
	'eta': 0.3, 
	'objective': 'multi:softmax', 
	'num_class' : 3,  
	'verbosity' : 0,
	'eval_metric' : 'mlogloss'}

dtrain = xgb.DMatrix(X_train, label = y_train_clf)
dtest = xgb.DMatrix(X_test, label = y_test_clf)
model = xgb.train(param, dtrain, 1000, 
	[(dtrain, 'train'), (dtest, 'eval')],
	early_stopping_rounds = 10)
BT = Classifier(model)
y_pred = BT.clf.predict(dtest)
xgb.plot_importance(BT.clf)
BT.evaluate(y_test_clf, y_pred)
plt.show()


#RF
print("Random Forest")
RF = Classifier(ensemble.RandomForestClassifier())
RF.fit(X_train, y_train_clf)
y_pred = RF.predict(X_test)
RF.evaluate(y_test_clf, y_pred)

#NB
print("Naive Bayes")
NB = Classifier(naive_bayes.GaussianNB())
NB.fit(X_train, y_train_clf)
y_pred = NB.predict(X_test)
NB.evaluate(y_test_clf, y_pred)
# SVM
print("Support vector machine")
SVM = Classifier(svm.SVC())
SVM.fit(X_train, y_train_clf)
y_pred = SVM.predict(X_test)
SVM.evaluate(y_test_clf, y_pred)

# MLP
print("Multi Perceptron Layer (NN)")
NN = Classifier(neural_network.MLPClassifier(hidden_layer_sizes = (100,), random_state=1, max_iter=300, alpha = 0.001))
NN.fit(X_train, y_train_clf)
y_pred = NN.predict(X_test)
NN.evaluate(y_test_clf, y_pred)

