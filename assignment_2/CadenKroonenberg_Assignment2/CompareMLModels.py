# Author: Caden Kroonenberg
# Date: 08-31-21

# load libraries
import numpy as np
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

def print_results(true, predicted):
    print('Accuracy Score:')
    print(accuracy_score(true, predicted))
    print('Confusion Matrix:')
    print(confusion_matrix(true, predicted))

# load dataset
url = "iris.csv"
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

# linear regression
lr_model = LinearRegression()
lr_model.fit(X_trainFold1, y_trainFold1)
lr_pred_fold1 = lr_model.predict(X_testFold1)
lr_pred_fold1 = lr_pred_fold1.round()
lr_model.fit(X_trainFold2, y_trainFold2)
lr_pred_fold2 = lr_model.predict(X_testFold2)
lr_pred_fold2 = lr_pred_fold2.round()
lr_y_pred = np.concatenate([lr_pred_fold1, lr_pred_fold2])
print('\nLINEAR REGRESSION')
print_results(y_true, lr_y_pred)

# deg. 2 polynomial regression
# transform data
transformer = PolynomialFeatures(degree=2, include_bias=False)
transformed_x_train_1 = transformer.fit_transform(X_trainFold1)
transformed_x_train_2 = transformer.fit_transform(X_trainFold2)
transformed_x_test_1 = transformer.fit_transform(X_testFold1)
transformed_x_test_2 = transformer.fit_transform(X_testFold2)

d2_pr_model = LinearRegression()

d2_pr_model.fit(transformed_x_train_1, y_trainFold1)
d2_pr_pred_fold1 = d2_pr_model.predict(transformed_x_test_1)
d2_pr_pred_fold1 = d2_pr_pred_fold1.round()

d2_pr_model.fit(transformed_x_train_2, y_trainFold2)
d2_pr_pred_fold2 = d2_pr_model.predict(transformed_x_test_2)
d2_pr_pred_fold2 = d2_pr_pred_fold2.round()

d2_pr_y_pred = np.concatenate([d2_pr_pred_fold1, d2_pr_pred_fold2])
# round misclassifications
for i in range(0, 150):
    if (d2_pr_y_pred[i] > 2):
        d2_pr_y_pred[i] = 2
    elif (d2_pr_y_pred[i] < 0):
        d2_pr_y_pred[i] = 0
d2_pr_y_pred = np.around(d2_pr_y_pred)
print('\nDEG. 2 POLYNOMIAL REGRESSION')
print_results(y_true, d2_pr_y_pred)

# deg. 3 polynomial regression
# transform data
transformer = PolynomialFeatures(degree=3, include_bias=False)
transformed_x_train_1 = transformer.fit_transform(X_trainFold1)
transformed_x_train_2 = transformer.fit_transform(X_trainFold2)
transformed_x_test_1 = transformer.fit_transform(X_testFold1)
transformed_x_test_2 = transformer.fit_transform(X_testFold2)

d2_pr_model = LinearRegression()

d2_pr_model.fit(transformed_x_train_1, y_trainFold1)
d2_pr_pred_fold1 = d2_pr_model.predict(transformed_x_test_1)
d2_pr_pred_fold1 = d2_pr_pred_fold1.round()

d2_pr_model.fit(transformed_x_train_2, y_trainFold2)
d2_pr_pred_fold2 = d2_pr_model.predict(transformed_x_test_2)
d2_pr_pred_fold2 = d2_pr_pred_fold2.round()

d2_pr_y_pred = np.concatenate([d2_pr_pred_fold1, d2_pr_pred_fold2])
# round misclassifications
for i in range(0, 150):
    if (d2_pr_y_pred[i] > 2):
        d2_pr_y_pred[i] = 2
    elif (d2_pr_y_pred[i] < 0):
        d2_pr_y_pred[i] = 0
d2_pr_y_pred = np.around(d2_pr_y_pred)
print('\nDEG. 3 POLYNOMIAL REGRESSION')
print_results(y_true, d2_pr_y_pred)

# naive-baysian
nb_model = GaussianNB()
nb_model.fit(X_trainFold1, y_trainFold1)
nb_pred_fold1 = nb_model.predict(X_testFold1)
nb_model.fit(X_trainFold2, y_trainFold2)
nb_pred_fold2 = nb_model.predict(X_testFold2)
nb_y_pred = np.concatenate([nb_pred_fold1, nb_pred_fold2])
print('\nNAIVE BAYSIAN')
print_results(y_true, nb_y_pred)

# kNN
nb_model = KNeighborsClassifier()
nb_model.fit(X_trainFold1, y_trainFold1)
nb_pred_fold1 = nb_model.predict(X_testFold1)
nb_model.fit(X_trainFold2, y_trainFold2)
nb_pred_fold2 = nb_model.predict(X_testFold2)
nb_y_pred = np.concatenate([nb_pred_fold1, nb_pred_fold2])
print('\nK-NEIGHBORS')
print_results(y_true, nb_y_pred)

# LDA
nb_model = LinearDiscriminantAnalysis()
nb_model.fit(X_trainFold1, y_trainFold1)
nb_pred_fold1 = nb_model.predict(X_testFold1)
nb_model.fit(X_trainFold2, y_trainFold2)
nb_pred_fold2 = nb_model.predict(X_testFold2)
nb_y_pred = np.concatenate([nb_pred_fold1, nb_pred_fold2])
print('\nLINEAR DISCRIMINANT ANALYSIS')
print_results(y_true, nb_y_pred)

# QDA
nb_model = QuadraticDiscriminantAnalysis()
nb_model.fit(X_trainFold1, y_trainFold1)
nb_pred_fold1 = nb_model.predict(X_testFold1)
nb_model.fit(X_trainFold2, y_trainFold2)
nb_pred_fold2 = nb_model.predict(X_testFold2)
nb_y_pred = np.concatenate([nb_pred_fold1, nb_pred_fold2])
print('\nQUADRATIC DISCRIMINANT ANALYSIS')
print_results(y_true, nb_y_pred)
