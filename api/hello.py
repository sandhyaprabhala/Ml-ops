from flask import Flask
from flask import request
import os
import numpy as np
app = Flask(__name__)

best_model_folder = './models/test_0.15_val_0.15_hyperparameter_0.001_i_2'

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/predict", methods=['POST'])
def predict():
        clf = load(os.path.join(best_model_folder,"models.joblib"))
        input_json = request.json
        image = input_json['image']
        print(image)
        image = np.array(image).reshape(1,-1)
        predicted = clf.predict(image)
        return str(predicted[0])

