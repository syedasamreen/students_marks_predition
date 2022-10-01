import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib
app = Flask(__name__)
model = joblib.load('LR_studentsMarks_pred.pkl')
df = pd.DataFrame()
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/Predict',method = ['POST'])
def predict():
    global  df

    input_feature = [int(x) for x in request.form.values()]
    feature_value = np.array(input_feature)

    # validate input hours
    if input_feature[0] < 0 or input_feature[0]>24:
        return render_template('index,html',prediction_text = "Please a valid hour")
