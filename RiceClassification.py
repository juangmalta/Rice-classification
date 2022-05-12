#1. Loading libraries
from re import X
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

#2. Load the data
data = pd.read_csv("./riceClassification.csv", index_col='id')

#3. Clean the data
data = data.dropna()

#4. Split the data into training and testing sets
X = data.drop('Class', axis=1)
Y = data['Class']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
clf = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, Y_train)

#5. Evaluate the model
print(clf.score(X_train, Y_train))
print(clf.score(X_test, Y_test))

#6. Confusion Matrix
y_pred = clf.predict(X_train)
cnf_matrix = metrics.confusion_matrix(Y_train, y_pred)
print(cnf_matrix)
y_pred_test = clf.predict(X_test)
cnf_matrix_test = metrics.confusion_matrix(Y_test, y_pred_test)
print(cnf_matrix_test)

#7. Plot the decision boundary
ax = plt.axes()
plt.scatter(X_train['Area'], X_train['Perimeter'], c=Y_train, cmap='autumn', label='Training data')
cb = plt.colorbar(label='Rice Type', ticks=[0, 1])
cb.ax.set_yticklabels(['Gonen', 'Jasmine'])
plt.xlabel('Area')
plt.ylabel('Perimeter')
plt.show()