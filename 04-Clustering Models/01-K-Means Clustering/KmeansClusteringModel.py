#Created on 05/05/2021
#@author saugatkafley

#Importing Libraires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Import Dataset
dataset= pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,3:5].values

#Using Elbow Method to find the optimum clusters 
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans = KMeans (n_clusters = i,init = 'k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
'''
plt.plot(range(1,11), wcss)
plt.title("The Elbow Method")
plt.xlabel("No of Clusters")
plt.ylabel("WCSS")
plt.show()
'''
#Traning dataset with 5 clusters 
kmeans = KMeans (n_clusters = 5,init = 'k-means++', random_state= 42)
y_kmeans = kmeans.fit_predict(X)

#Visualizing the clusters
plt.scatter(X[y_kmeans ==0,0],X[y_kmeans ==0,1],s=100,c='red',label = 'Cluster 1')
plt.scatter(X[y_kmeans ==1,0],X[y_kmeans ==1,1],s=100,c='blue',label = 'Cluster 2')
plt.scatter(X[y_kmeans ==2,0],X[y_kmeans ==2,1],s=100,c='black',label = 'Cluster 3')
plt.scatter(X[y_kmeans ==3,0],X[y_kmeans ==3,1],s=100,c='green',label = 'Cluster 4')
plt.scatter(X[y_kmeans ==4,0],X[y_kmeans ==4,1],s=100,c='yellow',label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1] , s=300, c='orange', label = 'Centers')
plt.title("Kmeasn clustering")
plt.legend()
plt.xlabel("Annual Income")
plt.ylabel("data")
plt.show()