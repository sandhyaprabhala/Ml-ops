from sklearn import datasets, svm
import matplotlib.pyplot as plt
from skimage.transform import rescale
import os
import utils


#---------data creation---------
digits = datasets.load_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' % label)
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

test_size = 0.3
gamma = [1e-05,0.0001,0.001,0.01]
for gmvalue in gamma:
    clf = svm.SVC(gamma=gmvalue)

    #Train-Validation-Test split
    X_train, X_test, X_val, y_train,y_test,y_val = utils.create_splits(data,digits.target,test_size)
    clf.fit(X_train, y_train)
    train_metrics = utils.test(clf,X_train,y_train)

#------------TODO - 1-----------
    def test_model_writing():

        #check the existance of the model.
        expected_path = utils.model_path(test_size,gmvalue)
        print(expected_path)
        assert os.path.exists(expected_path)

#------------TODO - 2-----------
    def test_small_data_overfit_checking():

        #throw away the models that yield random-like performance.
        assert train_metrics['acc'] > 0.11
        assert train_metrics['f1'] > 0.11
        