#k-fold Cross Validation

import numpy as np
import pandas as pd

dataset_X= pd.read_csv('train\X_train.txt',delim_whitespace=True, index_col=False, header=None)
X=dataset_X.iloc[:,:].values

dataset_Y=pd.read_csv('train\y_train.txt', header=None)
Y=dataset_Y.iloc[:,:].values


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.05, random_state=0,shuffle=True)

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
X_train=sc_x.fit_transform(X_train)
X_test=sc_x.transform(X_test)

from sklearn.linear_model import LogisticRegression
LR=LogisticRegression(random_state=0,solver='newton-cg', C=10,n_jobs=-1)
LR.fit(X_train,Y_train)

Y_pred=LR.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,Y_pred)

accuracy=LR.score(X_test,Y_test)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = LR, X = X_train, y = Y_train, cv = 10)
accuracies.mean()

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'solver': ['liblinear']},
              {'C': [1, 10, 100, 1000], 'solver': ['newton-cg']},
              {'C': [1,10,100,1000], 'solver' : ['sag']},
              {'C': [1, 10, 100, 1000], 'solver' : ['saga']}]
grid_search = GridSearchCV(estimator=LR,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X, Y)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

#Testing on given test set

dT_X=pd.read_csv('test\X_test.txt', delim_whitespace=True, index_col=False, header=None)
X_given=dT_X.iloc[:,:].values

dT_Y=pd.read_csv('test\Y_test.txt',header=None)
Y_given=dT_Y.iloc[:,:].values

X_given=sc_x.transform(X_given)
Y_pred_on_given=LR.predict(X_given)

realcm=confusion_matrix(Y_given,Y_pred_on_given)
final_accuracy=LR.score(X_given,Y_given)

