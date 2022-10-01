# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

## Load dataset
df = pd.read_csv("student_marks_hours.csv")
## Fill the missing values
df.fillna(df.mean(),inplace=True)

# Defining Feature and predicting variable.
X = df.drop("student_marks",axis="columns")
y = df.drop("study_hours",axis="columns")

# Split the data into train and split test.
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Define a suitable model to the data.
model = LinearRegression()

# Fit the model on train data.
model.fit(X_train,y_train)

# Predict the model on test data.
y_pred = model.predict(X_test)

# Check the score on test data.
Score = model.score(X_test,y_test)

# Finally save the model.
joblib.dump(model,'LR_studentsMarks_pred.pkl')

# valuavting the saved model.
saved_model = joblib.load('LR_studentsMarks_pred.pkl')
print(saved_model.predict([[5]])[0][0])