"""
================================
Recognizing hand-written digits
================================
This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.
"""

print(__doc__)

import random
from sklearn import datasets, svm
from skimage import data, color
import matplotlib.pyplot as plt
from skimage.transform import rescale
import numpy as np
from joblib import dump, load
import os
import utils
from sklearn import tree
import statistics as st

digits = datasets.load_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' % label)

#classification

#flatten the image

measures_Train_A = []
measures_Dev_A = []
measures_Test_A = []

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

for run in range(3):
    
    test_size = 0.3
    print("Run: {}".format(run+1))
    print("\nTrain:Dev:Test\t\tBest C\t\tBest Gamma\tA_Train_acc\tA_Dev_acc\tA_Test_acc\n")
   
    #Train-Validation-Test split
    X_train, X_test, X_val, y_train,y_test,y_val = utils.create_splits(data,digits.target,test_size)
        
    c_list = [0.01,1,10,100]
    gammas_list = [1e-05,0.0001,0.001,0.01]

    for i in range(3):
        model_candidates_A = []
        c_iter = random.choice(c_list)
        gmvalue = random.choice(gammas_list)

        #Create a classifier: a support vector classifier
        clf_A = svm.SVC(gamma=gmvalue, C=c_iter)

        #Learn the digits on the train data
        clf_A.fit(X_train, y_train)

        #Predict digits on the validation data
        metrics_train_A = utils.test(clf_A,X_train,y_train)
        metrics_dev_A = utils.test(clf_A,X_val,y_val)
        metrics_test_A = utils.test(clf_A,X_test,y_test)
        candidate_A ={"model" : clf_A,"acc_train" : metrics_train_A['acc'],"acc_valid" : metrics_dev_A['acc'],"acc_test" : metrics_test_A['acc'],"f1_valid" : metrics_dev_A['f1'],"c": c_iter, "gamma" : gmvalue}

        model_candidates_A.append(candidate_A)


        output_folder_A = utils.model_path(test_size,c_iter,gmvalue,run,i)
        os.mkdir(output_folder_A)

        dump(clf_A, os.path.join(output_folder_A,"models.joblib"))

        #loading the best model
        #predicting the test data on the best hyperparameter

        max_candidate_A = max(model_candidates_A, key = lambda x :x["f1_valid"])
        best_gamma = max_candidate_A["gamma"]
        best_c = max_candidate_A["c"]
        
        best_model_folder_A = "/home/sandhya/Ml-ops-repo/Ml-ops/mnist/models/test_{}_val_{}_c_{}_gamma_{}_run_{}_i_{}".format((test_size/2),(test_size/2),best_c,best_gamma,run,i)
        clf_A = load(os.path.join(best_model_folder_A,"models.joblib"))


        #printing accuracies
        metrics_train_A = utils.test(clf_A,X_train,y_train)
        metrics_dev_A = utils.test(clf_A,X_val,y_val)
        metrics_test_A = utils.test(clf_A,X_test,y_test)
        measures_Train_A.append(metrics_train_A['acc'])
        measures_Dev_A.append(metrics_dev_A['acc'])
        measures_Test_A.append(metrics_test_A['acc'])
        
        print("{} : {} : {}  \t{}\t\t{}\t\t{}\t\t{}\t\t{}".format((1-test_size),(test_size/2),(test_size/2),best_c,best_gamma,round(metrics_train_A['acc'],4),round(metrics_dev_A['acc'],4),round(metrics_test_A['acc'],4)))

    print("\nMean for clf_A (SVM) for Run {}: ".format(run+1))

    print("\nMean for Train: {}".format(st.mean(measures_Train_A)))
    print("Mean for Dev: {}".format(st.mean(measures_Dev_A)))
    print("Mean for Test: {}".format(st.mean(measures_Test_A)))


