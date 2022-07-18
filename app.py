import numpy as np
import pandas as pd
from flask import Flask, request, render_template,jsonify
from flask_cors import CORS, cross_origin
from sklearn import preprocessing
import pickle
import json

app = Flask(__name__)
cors = CORS(app)
model = pickle.load(open('model.pkl', 'rb'))

# @app.route('/')
# def home():
#     return render_template('index.html')
@app.route('/test',methods=['GET'])
def hello():
    if request.method=='GET':
        return 'Hello, World!'

@app.route('/predict',methods=['POST','GET'])
def predict():
    extract=json.loads(request.data.decode())
    prediction = model.predict(extract['data'])
    output = (prediction[0])
    print(output)
    if output == 1:
        text = "yes"
    else:
        text = "no"

    return jsonify({'response': text})


if __name__ == "__main__":
    app.run(debug=True,port=9090)