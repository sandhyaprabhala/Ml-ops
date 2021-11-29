import numpy as np
from numpy.lib.arraysetops import unique
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix

from joblib import load,dump
import os

digits = load_digits()

n_sample = len(digits.images)
data = digits.images.reshape((n_sample,-1))

# 80:20:20 for train:validation:test Split
train_data_size = 0.8
test_data_size = 0.1
valid_data_size = 0.1
train_x,test_x,train_y,test_y = train_test_split(data,digits.target,train_size=train_data_size)
val_x,test_x,val_y,test_y = train_test_split(test_x,test_y,train_size=(valid_data_size/(valid_data_size + test_data_size)))
#print(len(train_x),len(val_x),len(test_x))

def create_train_test_split(X,Y,train_size):
    #test_size = 1-train_size
    #assert train_size + test_size == 1
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=train_size)
    return X_train,X_test,Y_train,Y_test

def get_SVM_metrics(train_X,train_Y,test_X,test_Y,hyperparameter):
    clf = SVC(gamma=hyperparameter)
    clf.fit(train_X,train_Y)
    predict = clf.predict(test_X)
    acc = accuracy_score(y_true=test_Y,y_pred=predict)
    f1 = f1_score(y_true=test_Y,y_pred=predict,average='macro')
    return {'acc':acc,'f1':f1},clf

def get_test_metrics(model):
    predict = model.predict(test_x)
    acc = accuracy_score(y_true=test_y,y_pred=predict)
    f1 = f1_score(y_true=test_y,y_pred=predict,average='macro')
    return {'acc':acc,'f1':f1},predict

train_split_ratio = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
hyperparameter_svm = [0.00001,0.0001,0.001,0.1,1,2,5,10,20]
collection = []
for train_ratio in train_split_ratio:
    model_collection = []
    if train_ratio != 1:
        X_train,_,Y_train,_ = create_train_test_split(X = train_x,Y = train_y,train_size=train_ratio) # Spliting training data into various splits
    else:
        X_train,Y_train = train_x,train_y
    (unique,counts) = np.unique(Y_train,return_counts=True)
    freq = np.asarray((unique,counts)).T
    #print(train_ratio," Y values",freq)
    for gamma_val in hyperparameter_svm:
        # Using Validation data to evaluate a hyperparameter for a specific train split
        val_metrics,model = get_SVM_metrics(train_X = X_train,train_Y=Y_train,test_X=val_x,test_Y = val_y,hyperparameter=gamma_val)
        if val_metrics['f1'] >= 0.4:
            if val_metrics:
                candidate = {
                    "% of training data used":train_ratio*100,
                    "Validation F1 Score":val_metrics['f1'],
                    "Validation Accuracy":val_metrics['acc'],
                    "Hyperparameter":gamma_val
                }
            model_collection.append(candidate)
            #print(candidate)
            try:    # Preventing form the error if model is already present 
                output_folder = "./models/train_size_{}_svm_gamma_{}".format(train_ratio*100,gamma_val)
                os.mkdir(output_folder)
                dump(model,os.path.join(output_folder,'model.joblib'))
            except:
                pass
        else:
            print("Skipping for gamma ",gamma_val," and train size ",train_ratio," due to very low validation F1 Score i.e. ",val_metrics['f1'])

    max_valid_f1_score = max(model_collection,key=lambda x:x['Validation F1 Score'])
    best_model_folder = "./models/train_size_{}_svm_gamma_{}".format(max_valid_f1_score["% of training data used"],max_valid_f1_score["Hyperparameter"])
    path = os.path.join(best_model_folder,'model.joblib')
    best_model = load(path)
    test_metrics,prediction = get_test_metrics(best_model)
    print("\n")
    print("Test Metrics for ",max_valid_f1_score["% of training data used"]," ",max_valid_f1_score["Hyperparameter"]," ",test_metrics)
    plt.figure()
    cm = confusion_matrix(test_y,prediction)
    cm
    sns.heatmap(cm,annot=True)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    filename = "Confusion-Matrix for "+str(train_ratio*100)+" training data.png"
    plt.savefig(filename)
    del prediction
    if test_metrics:
        test_info = {
            "Test Accuracy":test_metrics['acc'],
            "Test F1 Score":test_metrics['f1']
        }
    max_valid_f1_score.update(test_info)
    collection.append(max_valid_f1_score)




df = pd.DataFrame(collection)

print(df)
plt.figure()
plt.plot(df['% of training data used'],df['Test F1 Score'])
plt.xlabel("Training data %")
plt.ylabel("Test F1 Score")
plt.title("Amount of Training data   Vs.  Test F1 Score")
plt.savefig("dataVSf1.png")
