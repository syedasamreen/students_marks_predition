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

@app.route('/predict',methods= ['POST'])
def predict():
    global df

    input_feature = [int(x) for x in request.form.values()]
    feature_value = np.array(input_feature)

    # validate input hours
    if input_feature[0] < 0 or input_feature[0]>24:
        return render_template('index,html',prediction_text = "Please Enter a valid hour")
    output = model.predict([feature_value])[0][0].round(2)
    df = pd.concat([df,pd.DataFrame({'Study Hours': input_feature,'Marks obtained':[output]})],ignore_index=True)
    print(df)
    df.to_csv('sample_data_from_ap.csv')
    return render_template('index.html',prediction_text = "You will get {}% marks, when you study {} hours".format(output,int(feature_value[0])))

if __name__ == "__main__":
    app.run()