import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_excel('Project_Data.xlsx')
x=dataset.iloc[:, 1:14].values

#Getting rid of categorical data of region...
category_Hot_Encoded=pd.get_dummies(x[:, 4])
x=np.append(x, category_Hot_Encoded, axis=1)
print(x)
x=np.delete(x, 4, 1)

#Getting rid of categorical data of Gender...
category_Hot_Encoded=pd.get_dummies(x[:, 1])
x=np.append(x, category_Hot_Encoded, axis=1)
print(x)
x=np.delete(x, 1, 1)

#Now we remove the female categorical data to avoid the categorical data trap
x=np.delete(x, 15, 1)

#Getting rid of categorical data of Goals...
category_Hot_Encoded=pd.get_dummies(x[:, 4])
x=np.append(x, category_Hot_Encoded, axis=1)
print(x)
x=np.delete(x, 4, 1)

#Getting rid of categorical data of Seats...
category_Hot_Encoded=pd.get_dummies(x[:, 7])
x=np.append(x, category_Hot_Encoded, axis=1)
print(x)
x=np.delete(x, 7, 1)# This removes the region coulumm which in turn would be replaced by category_Hot_Encoded

x_copy=x#This is used to maintain a x copy in object form
x=pd.DataFrame(x)

#Hobbies
clist = ['Game','Musical','Gym','Others']
for i in range(0, 325):
    list1 = x[3][i].split(', ')
    list2 = [0, 0, 0, 0]
    
    for j in range(0, 4):
        if clist[j] in list1:
            list2[j] = 1 
    x[3][i] = list2
    
    
tags = x[3].apply(pd.Series)
tags = tags.rename(columns = lambda a : 'tag_' + str(a))
x = pd.concat([x[:], tags[:]], axis=1)
x=pd.DataFrame(x)
x.rename(columns = {'tag_0':'Game'}, inplace = True)
x.rename(columns = {'tag_1':'Musical'}, inplace = True)
x.rename(columns = {'tag_2':'Gym'}, inplace = True)
x.rename(columns = {'tag_3':'Others'}, inplace = True)

#Delete column number 3(Hobbies) from the x...
x=x.drop(labels=3, axis=1)

#Nasha
clist = ['A','G','C','X','N']
for i in range(0, 325):
    list1 = x[6][i].split(', ')
    list2 = [0, 0, 0, 0, 0]
    for j in range(0, 5):
        if clist[j] in list1:
            list2[j] = 1    
    x[6][i] = list2
 
tags = x[6].apply(pd.Series)
tags = tags.rename(columns = lambda a : 'tag_' + str(a))
x = pd.concat([x[:], tags[:]], axis=1)
x=pd.DataFrame(x)
x.rename(columns = {'tag_0':'A'}, inplace = True)
x.rename(columns = {'tag_1':'G'}, inplace = True)
x.rename(columns = {'tag_2':'C'}, inplace = True)
x.rename(columns = {'tag_3':'X'}, inplace = True)
x.rename(columns = {'tag_4':'N'}, inplace = True)
x=x.drop(labels=6, axis=1)#Dropping Nasha column

#Music taste
clist = ['M1','M2','M3','M4','M5','M6','M7','M8','Others']
for i in range(0, 325):
    list1 = x[8][i].split(', ')
    list2 = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    for j in range(0, 9):
        if clist[j] in list1:
            list2[j] = 1
            
    x[8][i] = list2
    
    
tags = x[8].apply(pd.Series)
tags = tags.rename(columns = lambda a : 'tag_' + str(a))
x = pd.concat([x[:], tags[:]], axis=1)
x=pd.DataFrame(x)
x.rename(columns = {'tag_0':'M1'}, inplace = True)
x.rename(columns = {'tag_1':'M2'}, inplace = True)
x.rename(columns = {'tag_2':'M3'}, inplace = True)
x.rename(columns = {'tag_3':'M4'}, inplace = True)
x.rename(columns = {'tag_4':'M5'}, inplace = True)
x.rename(columns = {'tag_5':'M6'}, inplace = True)
x.rename(columns = {'tag_6':'M7'}, inplace = True)
x.rename(columns = {'tag_7':'M8'}, inplace = True)
x.rename(columns = {'tag_8':'Others'}, inplace = True)
x=x.drop(labels=8, axis=1)# This removes the Music coulumm
x=x.drop(labels=5, axis=1)# This removes the Friends coulumm
x=x.drop(labels=2, axis=1)# This removes the Music coulumm

"""

For usage:
0 - Scholar ID
1 - Income
2 - No. of friends
3 - Interaction with seniors
4-7 - Region
8 - Male
9-14 - Goals
15-17 - Seats
18-21 - Hobbies
22-26 - Nasha
27-35 - Music taste

"""

#Feature Scaling

##Scholar ID
y=dataset.iloc[:, 1:2].values
#type(y[1][1])
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
y = sc_X.fit_transform(y)
x=np.append(x, y, axis=1)
x=pd.DataFrame(x)
x=x.drop(labels=0, axis=1)


##Expenditure
y=dataset.iloc[:, 3:4].values
#type(y[1][1])
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
y = sc_X.fit_transform(y)
x=np.append(x, y, axis=1)
x=pd.DataFrame(x)
x=x.drop(labels=0, axis=1)

#Dimensionality Reduction
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
x = pca.fit_transform(x)
explained_variance = pca.explained_variance_ratio_

#Splitting the data set into train and test set...

x_train_cluster=x[65:325, :];
x_test_cluster=x[0:65, :];

#Model

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1]]
x_train_cluster, labels_true = make_blobs(n_samples=260, centers=centers, cluster_std=0.3,
                            random_state=0)

# Compute DBSCAN
db = DBSCAN(eps=0.3, min_samples=10).fit(x_train_cluster)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(x_train_cluster, labels))

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = x_train_cluster[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = x_train_cluster[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)
    


plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

#Random Forest

from sklearn.ensemble import RandomForestClassifier
y_labels=labels
y_train_cluster=y_labels
classifier=RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
classifier.fit(x_train_cluster, y_train_cluster)

#Predicting the group of test set...
y_pred=classifier.predict(x_test_cluster)

# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1]]
x_test_cluster, labels_true = make_blobs(n_samples=65, centers=centers, cluster_std=0.3,
                            random_state=0)

# Compute DBSCAN
db = DBSCAN(eps=0.4, min_samples=10).fit(x_test_cluster)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(x_test_cluster, labels))

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = x_test_cluster[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = x_test_cluster[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)


plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

