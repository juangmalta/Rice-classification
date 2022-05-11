from re import X
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
data = pd.read_csv("./riceClassification.csv", index_col='id')
print(data)

X = data.drop('Class', axis=1)
Y = data['Class']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
clf = LogisticRegression(random_state=0, max_iter=1000).fit(X, Y)
print(clf.score(X_train, Y_train))
print(clf.score(X_test, Y_test))

fig = plt.figure(figsize=(10,8))
plt.scatter(X_test['Area'],X_test['Perimeter'],marker='o',alpha=0.3,cmap='viridis')

plt.xlabel("feature 1")
plt.ylabel("feature 2")
plt.title('Random Classification Data with 2 classes')
plt.show()