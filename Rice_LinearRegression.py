#Linear Regression for the rice dataset using sklearn

#Libraries used:
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#Load the data
data = pd.read_csv("./riceClassification.csv",index_col='id')

#Clean the data and remove the null values
data=data.dropna()

#Y is the target variable and X is the predictor variable
y=data['Class']
x=data.drop(['Class','Extent'],axis=1)

#Split the data into training and testing data
testsize=0.2
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=testsize,random_state=42)
clf=LinearRegression().fit(x_train,y_train)

#Score the model
print("Accuracy on training data:",clf.score(x_train,y_train))
print("Accuracy on testing data:",clf.score(x_test,y_test))