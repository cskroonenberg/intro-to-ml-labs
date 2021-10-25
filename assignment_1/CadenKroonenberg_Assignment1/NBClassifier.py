# Author: Caden Kroonenberg
# Date: 08-31-21

# load libraries
import numpy as np
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

# load dataset
url = "iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class', 'class-num']
dataset = read_csv(url, skiprows=1, names=names)

# create arrays for features and classes
array = dataset.values
X = array[:,0:4]
y = array[:,4]

# split data into 2 folds for training and test
X_trainFold1, X_testFold1, y_trainFold1, y_testFold1 = train_test_split(X, y, test_size=0.50, random_state=1)
X_trainFold2 = X_testFold1
X_testFold2 = X_trainFold1
y_trainFold2 = y_testFold1
y_testFold2 = y_trainFold1

model = GaussianNB()
model.fit(X_trainFold1, y_trainFold1)
pred_fold1 = model.predict(X_testFold1)
model.fit(X_trainFold2, y_trainFold2)
pred_fold2 = model.predict(X_testFold2)

classes=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

# concatenate test data to true array and pred data to single pred array
y_true = np.concatenate([y_testFold1, y_testFold2])
y_pred = np.concatenate([pred_fold1, pred_fold2])

print('Accuracy Score:')
print(accuracy_score(y_true, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_true, y_pred))
print('Classification Report:')
print(classification_report(y_true, y_pred, target_names=classes))