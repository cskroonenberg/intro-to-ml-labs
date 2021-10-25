# Author: Caden Kroonenberg
# Date: 10-25-2021

import numpy as np
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, balanced_accuracy_score
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import RandomOverSampler 

def print_results(true, predicted):
    print('Confusion Matrix:')
    print(confusion_matrix(true, predicted))
    print('Accuracy Score:')
    print(accuracy_score(true, predicted))
    print('Class Balanced Accuracy')
    print(class_balanced_accuracy(true, predicted))
    print('Balanced Accuracy')
    print(balanced_accuracy(true, predicted))
    print('Scikit Balanced Accuracy Score')
    print(balanced_accuracy_score(true, predicted))

def class_balanced_accuracy(true, predicted):
    # Calculate recall and precision
    recall = recall_score(true, predicted, average=None)
    precision = precision_score(true, predicted, average=None)
    min_rp = [None]*len(recall)

    # Find minimum between precision and recall
    for i in range(len(recall)):
        min_rp[i] = min(recall[i], precision[i])

    # Take average of min values
    avg = sum(min_rp)/len(min_rp)
    return avg

def balanced_accuracy(true, predicted):
    # Calculate recall and specificity
    recall = recall_score(true, predicted, average=None)
    CM = confusion_matrix(true, predicted)
    FP = CM.sum(axis=0) - np.diag(CM)  
    FN = CM.sum(axis=1) - np.diag(CM)
    TP = np.diag(CM)
    TN = CM.sum() - (FP + FN + TP)
    S = TN/(TN+FP)

    avg = [None]*len(recall)
    # Calculate averages of recall and specificity for each class
    for i in range(len(recall)):
        avg[i] = (recall[i] + S[i])/2

    # Take average of average values
    avg = sum(avg)/len(avg)
    return avg

# load dataset
url = "imbalanced-iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class', 'class-num']
classes=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
dataset = read_csv(url, skiprows=1, names=names)

# create arrays for features and classes
array = dataset.values
X = array[:,0:4]
y = array[:,4]

# encode classes
encoded_classes = LabelEncoder()
encoded_classes.fit(y)
encoded_vals = encoded_classes.transform(y)

# split data into 2 folds for training and test
X_trainFold1, X_testFold1, y_trainFold1, y_testFold1 = train_test_split(X, encoded_vals, test_size=0.50, random_state=1)
X_trainFold2 = X_testFold1
X_testFold2 = X_trainFold1
y_trainFold2 = y_testFold1
y_testFold2 = y_trainFold1
y_true = np.concatenate([y_testFold1, y_testFold2])

# Neural Net
nn_model = MLPClassifier(max_iter=1000) # Increase maximum interation count to resolve convergence issue
nn_model.fit(X_trainFold1, y_trainFold1)
nn_pred_fold1 = nn_model.predict(X_testFold1)
nn_model.fit(X_trainFold2, y_trainFold2)
nn_pred_fold2 = nn_model.predict(X_testFold2)
nn_y_pred = np.concatenate([nn_pred_fold1, nn_pred_fold2])
print('\nNEURAL NET - Imbalanced Data Set')
print_results(y_true, nn_y_pred)
