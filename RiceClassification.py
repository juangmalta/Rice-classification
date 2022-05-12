from cProfile import label
from re import X
from statistics import linear_regression
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

#1. Load the data
data = pd.read_csv("./riceClassification.csv", index_col='id')

#2. Clean the data
data = data.dropna()

#3. Split the data into training and testing sets
X = data.drop(['Class','Extent'], axis=1)
Y = data['Class']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
clf = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, Y_train)

#4. Evaluate the model
print(clf.score(X_train, Y_train))
print(clf.score(X_test, Y_test))