"""
================================
Recognizing hand-written digits
================================
This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.
"""

print(__doc__)


# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports


# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm
from skimage import data, color
import matplotlib.pyplot as plt
from skimage.transform import rescale
import numpy as np
from joblib import dump, load
import os
import utils
from sklearn import tree

digits = datasets.load_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' % label)

#classification

#flatten the image
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

#test_sizes = [0.2, 0.3, 0.4]
#for test_size in test_sizes:
model_candidates = []
test_size = 0.3
#for gmvalue in [10 ** exponent for exponent in range(-7,0)]:
for max_depth in [5,6,7]:

  #Create a classifier: a support vector classifier
  
  #clf = svm.SVC(gamma=gmvalue)

  clf = tree.DecisionTreeClassifier(max_depth = max_depth)
    #Train-Validation-Test split
  X_train, X_test, X_val, y_train,y_test,y_val = utils.create_splits(data,digits.target,test_size)

    #Learn the digits on the train data
  clf.fit(X_train, y_train)

    #Predict digits on the validation data
  metrics_val = utils.test(clf,X_val,y_val)

    #throw away the models that yield random-like performance.

  #if metrics_val['acc'] < 0.11:
  #  print("{} not stored".format(max_depth))
  #  continue

  candidate ={"model" : clf,"acc_valid" : metrics_val['acc'],"f1_valid" : metrics_val['f1'], "depth" : max_depth}
  model_candidates.append(candidate)

  output_folder = utils.model_path(test_size,max_depth)
  os.mkdir(output_folder)
  dump(clf, os.path.join(output_folder,"models.joblib"))

    #loading the best model
    #predicting the test data on the best gamma
max_candidate = max(model_candidates, key = lambda x :x["f1_valid"])
best_depth = max_candidate["depth"]
#best_gamma = max_candidate["gamma"]
best_model_folder = "/home/sandhya/Ml-ops-repo/Ml-ops/mnist/models/test_{}_val_{}_hyperparameter_{}".format((test_size/2),(test_size/2),best_depth)

clf = load(os.path.join(best_model_folder,"models.joblib"))



#printing accuracies
metrics_test = utils.test(clf,X_test,y_test)

#print("\nBest Gamma is: {}\n".format(best_gamma))
print("\nBest Depth is: {}\n".format(best_depth))
print("Test Accuracy is {}".format(metrics_test['acc']))
print("f1_Score for Test Set is {}\n".format(metrics_test['f1']))
