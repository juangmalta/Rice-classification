#Libraries used:
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#Load the data
data = pd.read_csv("./riceClassification.csv",index_col='id')

data=data.dropna()

y=data['Class']
x=data.drop(['Class','Extent'],axis=1)

#Split the data into training and testing data
testsize=0.2
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=testsize,random_state=42)
clf=Perceptron(tol=1e-3,random_state=42).fit(x_train,y_train)
y_pred_train=clf.predict(x_train)
y_pred_test=clf.predict(x_test)

#Score the model
print("Accuracy on training data:",clf.score(x_train,y_train))
print("Accuracy on testing data:",clf.score(x_test,y_test))

#Confusion matrix on training data
print("Confusion matrix on training data:")
print(confusion_matrix(y_train,y_pred_train))
#Confusion matrix on testing data
print("Confusion matrix on testing data:")
print(confusion_matrix(y_test,y_pred_test))
