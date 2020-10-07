# %%

import numpy as np
import sklearn.svm
import joblib
from tabulate import tabulate
import sklearn.model_selection
from sklearn.preprocessing import StandardScaler

import misc_functions

# %%
# Data
predict_W1W2, predict_W1W2W3W4, train_test_W1W2, train_test_W1W2W3W4 = misc_functions.load_data()
target = train_test_W1W2W3W4['REDSHIFT']
target = target.apply(lambda x: x < 2.5)
features = train_test_W1W2W3W4.iloc[:, 2:10:2]
# %%

# Splitting data into training and target set 
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(features, target)

# Rescaling
X_train, y_train = X_train[:10000], y_train[:10000]
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

clf = sklearn.svm.SVC().fit(X_train, y_train)

#%%
scoring = {'acc': 'balanced_accuracy',
           'prec_macro': 'precision_macro',
           'rec_micro': 'recall_macro'}

clf_scores = sklearn.model_selection.cross_validate(clf, X_train, y_train, scoring=scoring, cv=5, return_train_score=True)

headers = ['kernal type'] + list(clf_scores.keys())
scores = [
    ['standard'] + list(clf_scores['test_acc']),
    ]

y_pred = clf.predict(X_test)
print(tabulate(scores, headers=headers))
print(sklearn.metrics.confusion_matrix(y_test, y_pred))
# %%
