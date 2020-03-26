#!/usr/bin/env python
# coding: utf-8

# In[1]:


# K-Means Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values


# In[3]:


# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 10), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[4]:


# Fitting K-Means to the dataset
best_cluster = 5
kmeans = KMeans(n_clusters = best_cluster, init = 'k-means++', random_state = 0)
pred = kmeans.fit_predict(X)

# Visualising the clusters
plt.rcParams["figure.figsize"] = (10,5)
colors = ["black", "blue", "green", "cyan", "yellow"]
for i in range(best_cluster):
    plt.scatter(X[pred == i, 0], X[pred == i, 1], c = colors[i], label = 'Cluster '+str(i+1))
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c = 'red', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


# In[ ]:




