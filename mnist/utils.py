#predefined functions
from sklearn.model_selection import train_test_split
from sklearn import datasets, svm, metrics
import os
from sklearn import tree

def create_splits(data,target,test_size):

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size= test_size, random_state=1)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=1)
    
    return X_train, X_test, X_val, y_train,y_test,y_val

def test(clf,X,y):

    m = dict()
    predicted = clf.predict(X)
    acc_1 = metrics.accuracy_score(y_pred=predicted, y_true=y)
    f1_score = metrics.f1_score(y_pred=predicted, y_true=y, average='macro')
    m["acc"] = acc_1
    m["f1"] = f1_score

    return m

def model_path(test_size,value,i):
    
    output_folder = "/home/sandhya/Ml-ops-repo/Ml-ops/mnist/models/test_{}_val_{}_hyperparameter_{}_i_{}".format((test_size/2),(test_size/2),value,i)
    return output_folder
