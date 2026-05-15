from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

file_path = os.path.join(os.path.dirname(__file__), "data", "student-mat.csv")
df = pd.read_csv(file_path, sep=';')

df = df.drop_duplicates()

features = ['studytime', 'failures', 'absences', 'G1', 'G2']
X = df[features]
y = df['G3']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    studytime = float(request.form['studytime'])
    failures = float(request.form['failures'])
    absences = float(request.form['absences'])
    G1 = float(request.form['G1'])
    G2 = float(request.form['G2'])

    input_data = np.array([[studytime, failures, absences, G1, G2]])

    predicted_g3 = model.predict(input_data)[0]
    percentage = (predicted_g3 / 20) * 100

    if percentage >= 85:
        gpa = 4.00
    elif percentage >= 80:
        gpa = 3.70
    elif percentage >= 75:
        gpa = 3.30
    elif percentage >= 70:
        gpa = 3.00
    elif percentage >= 65:
        gpa = 2.70
    elif percentage >= 60:
        gpa = 2.30
    elif percentage >= 55:
        gpa = 2.00
    elif percentage >= 50:
        gpa = 1.70
    elif percentage >= 45:
        gpa = 1.30
    elif percentage >= 40:
        gpa = 1.00
    else:
        gpa = 0.00

    return render_template(
        'index.html',
        prediction=round(predicted_g3, 2),
        gpa=format(gpa, ".2f")
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)