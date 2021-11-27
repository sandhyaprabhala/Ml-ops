"""
================================
Recognizing hand-written digits
================================
This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.
"""

print(__doc__)

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

measures_A = []
measures_B = []

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

test_size = 0.3

print("\nTrain:Test:valid\tBest Gamma\tBest Depth\tA_Test_acc\tB_Test_acc\tA_f1_score\tB_f1_score\n")
#Train-Validation-Test split
X_train, X_test, X_val, y_train,y_test,y_val = utils.create_splits(data,digits.target,test_size)
       
depths = [5,6,7,8]
gammas = [1e-05,0.0001,0.001,0.01]
for i in range(5):
    model_candidates_A = []
    model_candidates_B = []
    for gmvalue,max_depth in zip(gammas,depths):

        #Create a classifier: a support vector classifier
        clf_A = svm.SVC(gamma=gmvalue)
        clf_B = tree.DecisionTreeClassifier(max_depth = max_depth)

        #Learn the digits on the train data
        clf_A.fit(X_train, y_train)
        clf_B.fit(X_train, y_train)

        #Predict digits on the validation data
        metrics_val_A = utils.test(clf_A,X_val,y_val)
        metrics_val_B = utils.test(clf_B,X_val,y_val)

        candidate_A ={"model" : clf_A,"acc_valid" : metrics_val_A['acc'],"f1_valid" : metrics_val_A['f1'], "gamma" : gmvalue}
        candidate_B ={"model" : clf_B,"acc_valid" : metrics_val_B['acc'],"f1_valid" : metrics_val_B['f1'], "depth" : max_depth}
        model_candidates_A.append(candidate_A)
        model_candidates_B.append(candidate_B)

        output_folder_A = utils.model_path(test_size,gmvalue,i)
        os.mkdir(output_folder_A)
        output_folder_B = utils.model_path(test_size,max_depth,i)
        os.mkdir(output_folder_B)
        dump(clf_A, os.path.join(output_folder_A,"models.joblib"))
        dump(clf_B, os.path.join(output_folder_B,"models.joblib"))

    #loading the best model
    #predicting the test data on the best hyperparameter

    max_candidate_A = max(model_candidates_A, key = lambda x :x["f1_valid"])
    best_gamma = max_candidate_A["gamma"]
    
    best_model_folder_A = "/home/sandhya/Ml-ops-repo/Ml-ops/mnist/models/test_{}_val_{}_hyperparameter_{}_i_{}".format((test_size/2),(test_size/2),best_gamma,i)
    clf_A = load(os.path.join(best_model_folder_A,"models.joblib"))


    max_candidate_B = max(model_candidates_B, key = lambda x :x["f1_valid"])
    best_depth = max_candidate_B["depth"]
    
    best_model_folder_B = "/home/sandhya/Ml-ops-repo/Ml-ops/mnist/models/test_{}_val_{}_hyperparameter_{}_i_{}".format((test_size/2),(test_size/2),best_depth,i)
    clf_B = load(os.path.join(best_model_folder_B,"models.joblib"))



    #printing accuracies
    metrics_test_A = utils.test(clf_A,X_test,y_test)
    metrics_test_B = utils.test(clf_B,X_test,y_test)
    
    measures_A.append(metrics_test_A['acc'])
    measures_B.append(metrics_test_B['acc'])

    print("{} : {} : {}  \t{}\t\t{}\t\t{}\t\t{}\t\t{}\t\t{}".format((1-test_size),(test_size/2),(test_size/2),best_gamma,best_depth,round(metrics_test_A['acc'],4),round(metrics_test_B['acc'],4),round(metrics_test_A['f1'],4),round(metrics_test_B['f1'],4)))

print("\nMean and Standard Deviation for clf_A (SVM):")
print("\nMean: {}".format(st.mean(measures_A)))
print("Standard Deviation: {}\n".format(st.stdev(measures_A)))

print("\nMean and Standard Deviation for clf_B (Decision Tree):")
print("\nMean: {}".format(st.mean(measures_B)))
print("Standard Deviation: {}\n".format(st.stdev(measures_B)))