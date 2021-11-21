from flask import Flask
from flask import request
from joblib import load
import os
import numpy as np
app = Flask(__name__)


best_model_folder = '/home/sandhya/Ml-ops-repo/Ml-ops/mnist/models/test_0.15_val_0.15_hyperparameter_8_i_2/models.joblib'
clf = load(best_model_folder)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/svm_decision_tree", methods=['POST'])
def predict():
        input_json = request.json
        image = input_json['image']
        print(image)
        image = np.array(image).reshape(1,-1)
        predicted = clf.predict(image)
        return str(predicted[0])

