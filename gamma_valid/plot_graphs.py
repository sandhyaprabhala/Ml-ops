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
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from skimage import data, color
from skimage.transform import rescale
import numpy as np
from joblib import dump, load
import os


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
for gmvalue in [10 ** exponent for exponent in range(-7,0)]:
  #Create a classifier: a support vector classifier
  clf = svm.SVC(gamma=gmvalue)

  #Train-Validation-Test split
  X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size= test_size, random_state=1)
  X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=1)

  #Learn the digits on the train data
  clf.fit(X_train, y_train)

  #Predict the value of the digit on the validation
  predicted2 = clf.predict(X_val)

  acc = metrics.accuracy_score(y_pred=predicted2, y_true=y_val)
  f1_valid = metrics.f1_score(y_pred=predicted2, y_true=y_val, average='macro')

  #throw away the models that yield random-like performance.
  if acc < 0.11:
    print("{} not stored".format(gmvalue))
    continue
  candidate ={"model" : clf,"acc_valid" : acc,"f1_valid" : f1_valid,"gamma" : gmvalue,}
  model_candidates.append(candidate)

  #storing the model
  output_folder = "/home/sandhya/Ml-ops-repo/Ml-ops/gamma_valid/models/test_{}_val_{}_gamma_{}".format((test_size/2),(test_size/2),gmvalue)
  os.mkdir(output_folder)
  dump(clf, os.path.join(output_folder,"models.joblib"))

  #loading the best model
  #predicting the test data on the best gamma
max_candidate = max(model_candidates, key = lambda x :x["f1_valid"])
best_gamma = max_candidate["gamma"]
best_model_folder = "/home/sandhya/Ml-ops-repo/Ml-ops/gamma_valid/models/test_{}_val_{}_gamma_{}".format((test_size/2),(test_size/2),max_candidate["gamma"])

clf = load(os.path.join(best_model_folder,"models.joblib"))
predicted1 = max_candidate["model"].predict(X_test)


#printing accuracies
acc1 = metrics.accuracy_score(y_pred=predicted1, y_true=y_test)
f1_test = metrics.f1_score(y_pred=predicted1, y_true=y_test, average='macro')

print("\nBest Gamma is: {}\n".format(best_gamma))
print("Test Accuracy at {} is {}".format(best_gamma,acc1))
print("f1_Score for Test Set at {} is {}\n".format(best_gamma,f1_test))
