# Author: Caden Kroonenberg
# Date: November 8, 2021

import numpy as np
from numpy.core.fromnumeric import size
from pandas import read_csv
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from PlottingCode import plot_graph
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib import pyplot as plt
    
def maxSecondDerivative(data):
    maxDrv = 0
    maxIdx = 0
    for i in range(2,size(data)-1):                            # Ignore k = 2 which will always have largest second derivative by an overwhelming amount due to high reconstruction error at k = 1
        approx_SndDrv = (data[i+1] + data[i-1] - 2 * data[i])  # Approximate second derivative with central difference
        if maxDrv < approx_SndDrv: # Find largest second derivative; log second derivative and index
            maxDrv = approx_SndDrv
            maxIdx = i
    return maxIdx + 1 # Index i indicates k = i + 1

# load dataset
url = "iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class', 'class-num']
dataset = read_csv(url, skiprows=1, names=names)

# create arrays for features and classes
array = dataset.values
X = array[:,0:4]
y = array[:,4]

# k-Means Clustering, k = elbow_k

# Generate reconstruction error values for k = 1 to k = 20
recons_err_arr = np.empty(shape=(20))
for i in range(size(recons_err_arr)):
    kmeans = KMeans(n_clusters=i+1, random_state=0).fit(X)  # n_clusters = i+1 because of 0 indexing (i = 0 indicates k = 1)
    recons_err_arr[i] = kmeans.inertia_                     # Add reconstruction error value for k = i+1 to reconstruction error array

elbow_k = maxSecondDerivative(recons_err_arr) # Set elbow value to the reonstruction error with the highest second derivative (indicating highest curvature)

# Plot reconstruction error
plot_graph(recons_err_arr, 'Reconstruction Error')

kmeans = KMeans(n_clusters=elbow_k, random_state=0).fit(X)  # Generate k-means clustering for elbow_k clusters

truth = np.empty(shape=size(y)) # Encode labels from data set to numerical values
for i in range(size(y)):
    if y[i] == 'Iris-setosa':
        truth[i] = 0
    elif y[i] == 'Iris-versicolor':
        truth[i] = 1
    elif y[i] == 'Iris-virginica':
        truth[i] = 2

# Prep
k_labels = kmeans.labels_                   # Get cluster labels
k_labels_matched = np.empty_like(k_labels)  # Array to track matched cluster labels for samples

for k in np.unique(k_labels): # For each cluster label find and assign the best-matching truth label
    match_nums = [np.sum((k_labels==k)*(truth==t)) for t in np.unique(truth)]
    k_labels_matched[k_labels==k] = np.unique(truth)[np.argmax(match_nums)]

# Compute confusion matrix
cm = confusion_matrix(truth, k_labels_matched)
print('k-Means Clustering, k = elbow_k (Determined through maximum second derivative of reconstruction error values)')
print('Confusion Matrix:')
print(cm)
print('Accuracy Score:')
print(accuracy_score(truth, k_labels_matched))

# Plot confusion matrix
# plt.imshow(cm,interpolation='none',cmap='Blues')
# for (i, j), z in np.ndenumerate(cm):
#     plt.text(j, i, z, ha='center', va='center')
# plt.xlabel("kmeans label")
# plt.ylabel("truth label")
# plt.show()

# k-Means Clustering, k = 3
k = 3
kmeans = KMeans(n_clusters=elbow_k, random_state=0).fit(X)  # Generate k-means clustering for elbow_k clusters

# Prep
k_labels = kmeans.labels_                   # Get cluster labels
k_labels_matched = np.empty_like(k_labels)  # Array to track matched cluster labels for samples

for k in np.unique(k_labels): # For each cluster label find and assign the best-matching truth label
    match_nums = [np.sum((k_labels==k)*(truth==t)) for t in np.unique(truth)]
    k_labels_matched[k_labels==k] = np.unique(truth)[np.argmax(match_nums)]

# Compute confusion matrix
cm = confusion_matrix(truth, k_labels_matched)
print('\nk-Means Clustering, k = 3')
print('Confusion Matrix:')
print(cm)
print('Accuracy Score:')
print(accuracy_score(truth, k_labels_matched))

# Plot confusion matrix
# plt.imshow(cm,interpolation='none',cmap='Blues')
# for (i, j), z in np.ndenumerate(cm):
#     plt.text(j, i, z, ha='center', va='center')
# plt.xlabel("kmeans label")
# plt.ylabel("truth label")
# plt.show()

# Gaussian Mixture Models

# AIC
aic = np.empty(shape=(20,1))
for i in range(1,21):
    gm = GaussianMixture(n_components=i, random_state=0, covariance_type='diag').fit(X)
    aic[i-1] = gm.aic(X)

# Plot AIC
plot_graph(aic, 'AIC')

aic_elbow_k = 3 # An elbow value of 3 was obtained using the elbow method on the AIC graph

gm_aic = GaussianMixture(n_components=aic_elbow_k, random_state=0, covariance_type='diag').fit(X)

# Prep
gm_aic_pred = gm_aic.predict(X)
aic_labels_matched = np.empty_like(gm_aic_pred)

for k in np.unique(gm_aic_pred): # For each cluster label find and assign the best-matching truth label
    match_nums = [np.sum((gm_aic_pred==k)*(truth==t)) for t in np.unique(truth)]
    aic_labels_matched[gm_aic_pred==k] = np.unique(truth)[np.argmax(match_nums)]

# Compute confusion matrix
cm = confusion_matrix(truth, aic_labels_matched)
print('\nGMM, k determined from AIC')
print('Confusion Matrix:')
print(cm)
print('Accuracy Score:')
print(accuracy_score(truth, aic_labels_matched))

# Plot confusion matrix
# plt.imshow(cm,interpolation='none',cmap='Blues')
# for (i, j), z in np.ndenumerate(cm):
#     plt.text(j, i, z, ha='center', va='center')
# plt.xlabel("aic label")
# plt.ylabel("truth label")
# plt.show()

# BIC
bic = np.empty(shape=(20,1))
for i in range(1,21):
    gm = GaussianMixture(n_components=i, random_state=0, covariance_type='diag').fit(X)
    bic[i-1] = gm.bic(X)

# Plot BIC
plot_graph(bic, 'BIC')

bic_elbow_k = 4 # An elbow value of 4 was obtained using the elbow method on the BIC graph

gm_bic = GaussianMixture(n_components=bic_elbow_k, random_state=0, covariance_type='diag').fit(X)

# Prep
gm_bic_pred = gm_bic.predict(X)
bic_labels_matched = np.empty_like(gm_bic_pred)

# Compute confusion matrix
cm = confusion_matrix(truth, gm_bic_pred)
print('\nGMM, k determined from BIC')
print('Confusion Matrix:')
print(cm)
print('Accuracy Score:')
print("Cannot calculate Accuracy Score because the number of classes is not the same as the number of clusters")

# Plot confusion matrix
# plt.imshow(cm,interpolation='none',cmap='Blues')
# for (i, j), z in np.ndenumerate(cm):
#     plt.text(j, i, z, ha='center', va='center')
# plt.xlabel("bic label")
# plt.ylabel("truth label")
# plt.show()

# GMM, k = 3

k = 3

gm_aic = GaussianMixture(n_components=k, random_state=0, covariance_type='diag').fit(X)

# Prep
gm_aic_pred = gm_aic.predict(X)
aic_labels_matched = np.empty_like(gm_aic_pred)

for k in np.unique(gm_aic_pred): # For each cluster label find and assign the best-matching truth label
    match_nums = [np.sum((gm_aic_pred==k)*(truth==t)) for t in np.unique(truth)]
    aic_labels_matched[gm_aic_pred==k] = np.unique(truth)[np.argmax(match_nums)]

# Compute confusion matrix
cm = confusion_matrix(truth, aic_labels_matched)
print('\nGMM, k = 3')
print('Confusion Matrix:')
print(cm)
print('Accuracy Score:')
print(accuracy_score(truth, aic_labels_matched))
# Plot confusion matrix
# plt.imshow(cm,interpolation='none',cmap='Blues')
# for (i, j), z in np.ndenumerate(cm):
#     plt.text(j, i, z, ha='center', va='center')
# plt.xlabel("bic label")
# plt.ylabel("truth label")
# plt.show()