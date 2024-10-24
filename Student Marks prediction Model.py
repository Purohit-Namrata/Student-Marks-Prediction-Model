import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("C:/Users/BLAUPLUG/Documents/Python_programs/Student Marks Prediction Model/Student marks/Student_Marks.csv")
#print(data.head(10))

print(data.isnull().sum())

data["number_courses"].value_counts()

figure = px.scatter(data_frame=data, x = "number_courses", 
                    y = "Marks", size = "time_study", 
                    title="Number of Courses and Marks Scored")
figure.show()

correlation = data.corr()
print(correlation["Marks"].sort_values(ascending=False))

X = np.array(data[["time_study", "number_courses"]])
y = np.array(data["Marks"])
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))

# Features = [["time_study", "number_courses"]]
features = np.array([[4.508, 3]])
print(model.predict(features))
