from cProfile import label
from re import X
from matplotlib import cm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#1. Load the data
data = pd.read_csv("./riceClassification.csv", index_col='id')
print(data)

#2. Clean the data
data = data.dropna()

#3. Split the data into training and testing sets
X = data.drop('Class', axis=1)
Y = data['Class']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
clf = LogisticRegression(random_state=0, max_iter=1000).fit(X, Y)

#4. Evaluate the model
print(clf.score(X_train, Y_train))
print(clf.score(X_test, Y_test))

#5. Predict the class of a new sample
ax = plt.axes()
plt.scatter(X_train['Area'], X_train['Perimeter'], c=Y_train, cmap='autumn', label='Training data')
cb = plt.colorbar(label='Rice Type', ticks=[0, 1])
cb.ax.set_yticklabels(['Gonen', 'Jasmine'])
plt.ylabel('Perimeter')
plt.xlabel('Area')
plt.show()

ax = plt.axes()
plt.scatter(X_train['MajorAxisLength'], X_train['MinorAxisLength'], c=Y_train, cmap='autumn', label='Training data')
cb = plt.colorbar(label='Rice Type', ticks=[0, 1])
cb.ax.set_yticklabels(['Gonen', 'Jasmine'])
plt.ylabel('MinorAxisLength')
plt.xlabel('MajorAxisLength')
plt.show()

ax = plt.axes()
plt.scatter(X_test['Area'], X_test['Perimeter'], c=Y_test, cmap='autumn', label='Testing data')
cb = plt.colorbar(label='Rice Type', ticks=[0, 1])
cb.ax.set_yticklabels(['Gonen', 'Jasmine'])
plt.ylabel('Perimeter')
plt.xlabel('Area')
plt.show()

ax = plt.axes()
plt.scatter(X_test['MajorAxisLength'], X_test['MinorAxisLength'], c=Y_test, cmap='autumn', label='Testing data')
cb = plt.colorbar(label='Rice Type', ticks=[0, 1])
cb.ax.set_yticklabels(['Gonen', 'Jasmine'])
plt.ylabel('MinorAxisLength')
plt.xlabel('MajorAxisLength')
plt.show()

