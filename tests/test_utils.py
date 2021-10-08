from sklearn import datasets, svm
import matplotlib.pyplot as plt
from skimage.transform import rescale
import os
import utils


#---------data creation---------
digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
test_size = 0.3
gamma = [1e-05,0.0001,0.001,0.01]
for gmvalue in gamma:
    clf = svm.SVC(gamma=gmvalue)

    
#---------QUIZ - 2--------------
#-------test cases for the create_split------  
    def test_create_split():

        #Train-Validation-Test split
        X_train, X_test, X_val, y_train,y_test,y_val = utils.create_splits(data,digits.target,test_size)
        x = len(X_train)
        y = len(X_test)
        z = len(X_val)
        sum = x + y + z
        x1 = int((sum * 0.7))
        x2 = int((sum * 0.201))
        x3 = int((sum * 0.1))
        assert sum == n_samples
        assert x == x1
        assert y == x2
        assert z == x3
    
  