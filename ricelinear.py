import numpy as np
import pandas as pd

from sklearn.linear_model import Perceptron
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

x = pd.read_csv("./riceClassification.csv")

print(x.columns)

print(x.head())