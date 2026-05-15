# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score

# df = pd.read_csv('Student Performance UCI\data\student-mat.csv', sep=';')

# df = df.drop_duplicates()

# print(df.head())
# print(df.shape)
# print(df.columns)
# print(df.info())
# print(df.isnull().sum())
# print(df.duplicated().sum())
# print(df.describe())
# print(df.dtypes)

# categorical_columns = df.select_dtypes(include=['object']).columns
# print(categorical_columns)

# df_encoded = pd.get_dummies(df, drop_first=True)

# plt.figure(figsize=(6,4))
# plt.boxplot(df['absences'])
# plt.title("Absences Distribution")
# plt.grid(True)
# plt.show()

# print(df_encoded.corr()['G3'].sort_values(ascending=False))

# features = ['studytime', 'failures', 'absences', 'G1', 'G2']
# target = 'G3'

# X = df[features]
# y = df[target]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model = LinearRegression()
# model.fit(X_train, y_train)

# print('Model Trained Successfully')
# print("Intercept:", model.intercept_)

# prediction = model.predict(X_test)

# mse = mean_squared_error(y_test, prediction)
# rmse = np.sqrt(mse)
# r2 = r2_score(y_test, prediction)

# print("MSE:", mse)
# print("RMSE:", rmse)
# print("R2 Score:", r2)

# plt.figure(figsize=(6,5))
# plt.scatter(y_test, prediction, alpha=0.7, edgecolor='black')
# plt.xlabel("Actual G3")
# plt.ylabel("Predicted G3")
# plt.title("Actual vs Predicted Grades")
# plt.grid(True)
# plt.show()

# print("\nEnter Student Details for GPA Prediction:")

# studytime = float(input("Study Time: "))
# failures = float(input("Failures: "))
# absences = float(input("Absences: "))
# G1 = float(input("G1 Marks: "))
# G2 = float(input("G2 Marks: "))

# new_student = [[studytime, failures, absences, G1, G2]]

# predicted_g3 = model.predict(new_student)[0]

# gpa = (predicted_g3 / 20) * 4

# print("\nPredicted Final Marks (G3):", round(predicted_g3, 2))
# print("Predicted GPA (4.0 Scale):", round(gpa, 2))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('Student Performance UCI\\data\\student-mat.csv', sep=';')

df = df.drop_duplicates()

print(df.head())
print(df.shape)
print(df.columns)
print(df.info())
print(df.isnull().sum())
print(df.duplicated().sum())
print(df.describe())
print(df.dtypes)

categorical_columns = df.select_dtypes(include=['object']).columns
print(categorical_columns)

df_encoded = pd.get_dummies(df, drop_first=True)

plt.figure(figsize=(6,4))
plt.boxplot(df['absences'])
plt.title("Absences Distribution")
plt.grid(True)
plt.show()

print(df_encoded.corr()['G3'].sort_values(ascending=False))

features = ['studytime', 'failures', 'absences', 'G1', 'G2']
target = 'G3'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print('Model Trained Successfully')
print("Intercept:", model.intercept_)

prediction = model.predict(X_test)

mse = mean_squared_error(y_test, prediction)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, prediction)

print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

plt.figure(figsize=(6,5))
plt.scatter(y_test, prediction, alpha=0.7, edgecolor='black')
plt.xlabel("Actual G3")
plt.ylabel("Predicted G3")
plt.title("Actual vs Predicted Grades")
plt.grid(True)
plt.show()

print("\nEnter Student Details for GPA Prediction:")

studytime = float(input("Study Time: "))
failures = float(input("Failures: "))
absences = float(input("Absences: "))
G1 = float(input("G1 Marks: "))
G2 = float(input("G2 Marks: "))

new_student = [[studytime, failures, absences, G1, G2]]

predicted_g3 = model.predict(new_student)[0]

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

print("\nPredicted Final Marks (G3):", round(predicted_g3, 2))
print("Predicted Percentage:", round(percentage, 2), "%")
print("Predicted GPA (4.00 Scale):", format(gpa, ".2f"))