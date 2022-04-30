from flask import Flask, request, url_for, redirect, render_template, jsonify
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open("gb.pkl", "rb"))
sc = pickle.load(open("scalar.pkl", "rb"))
cols = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age',
        'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat']


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    print(request.form.to_dict())
    df = pd.DataFrame(columns=cols)
    for col in cols:
        df[col] = [request.form[col]]
    print(df.values)
    # df = pd.DataFrame(request.form.to_dict(), index=[0])
    # int_features = [int(x) for x in request.form.values()]
    # final = np.array(int_features)
    # final1 = pd.DataFrame([final], columns=cols)
    # print(final1.values)
    x = sc.transform(df.values)
    prediction = model.predict(x)[0]
    print(prediction)
    msg = 'Predicted House price will be '+ str(np.round(prediction*10000,2)) +"$."
    return render_template('index.html', pred=msg)


if __name__ == '__main__':
    app.run(debug=True)
