import numpy as np
import pandas as pd

from sklearn.linear_model import Perceptron
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

x = pd.read_csv("./riceClassification.csv")

print(x.columns)

print(x.head())

featuresdrop= ['id','Area','MajorAxisLength','MinorAxisLength','Eccentricity','ConvexArea','EquivDiameter','Extent','Perimeter','Roundness','AspectRation','Class']
x=x.drop(featuresdrop,axis=1)
x=x.dropna

y=x['Class']
x=x.drop('Class',axis=1)

testsize=0.2

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=testsize,random_state=42)
clf=Perceptron(tol=1e-3,random_state=None).fit(x_train,y_train)
y_pred_train=clf.predict(x_train)
y_pred_test=clf.predict(x_test)

print(clf.score(x_train,y_train))

print(clf.score(x_test,y_test))

print(confusion_matrix(y_train,y_pred_train))
print(confusion_matrix(y_test,y_pred_test))
