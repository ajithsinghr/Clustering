# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 12:25:19 2023

@author: ramav
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("D:\\Assignments\\clustering\\EastWestAirlines.csv")
df.head()
df.isnull().sum()
df.dtypes

df = df.drop("ID#",axis=1)
df.shape
df.columns

# histogram
plt.hist(df["Balance"])
plt.hist(df["Qual_miles"])
plt.hist(df["cc1_miles"])
plt.hist(df["cc2_miles"])
plt.hist(df["cc3_miles"])
plt.hist(df["Bonus_miles"])
plt.hist(df["Bonus_trans"])
plt.hist(df["Flight_miles_12mo"])
plt.hist(df["Flight_trans_12"])
plt.hist(df["Days_since_enroll"])


# BOX PLOT
plt.boxplot(df['Balance'])
plt.boxplot(df['Qual_miles'])
plt.boxplot(df['cc1_miles'])
plt.boxplot(df['cc2_miles'])
plt.boxplot(df['cc3_miles'])
plt.boxplot(df['Bonus_miles'])
plt.boxplot(df['Bonus_trans'])
plt.boxplot(df['Flight_miles_12mo'])
plt.boxplot(df['Days_since_enroll'])

# we found outliers in all the above variable expect days sinnce enroll
# 
cat_list=[]
num_list=[]


for i in df.columns:
    unique_values = len(df[i].unique())
    if unique_values<10:
        cat_list.append(i)
    else:
        num_list.append(i)
        
cat_list
num_list

# distributions of #num_list

k=1
plt.figure(figsize=(12,12))
plt.suptitle("distribution of numerical values")
for i in df.loc[:,num_list]:
    plt.subplot(4,2,k)
    sns.distplot(df[i])
    plt.title(i)
    plt.tight_layout()
    k+=1

# detection of outliers
for i in df.loc[:,num_list]:
    Q1 = df[i].quantile(0.25)
    Q3 = df[i].quantile(0.75)
    IQR = Q3-Q1
    up = Q3 + 1.5*IQR
    low = Q1 - 1.5*IQR

    if df[(df[i] > up) | (df[i] < low)].any(axis=None):
        print(i,"yes")
    else:
        print(i, "no")

# making boxplot in for loop uses to make easy box plots

k=1
plt.figure(figsize=(13,13))
plt.suptitle("Distribution of Outliers")

for i in df.loc[:,num_list]:
    plt.subplot(4,2,k)
    sns.boxplot(x = i, data = df.loc[:,num_list])
    plt.title(i)
    plt.tight_layout()
    k+=1

# treating outliers
out_list=["Bonus_trans","Flight_miles_12mo","Flight_trans_12"]

for i in df.loc[:,out_list]:
    Q1 = df[i].quantile(0.25)
    Q3 = df[i].quantile(0.75)
    IQR = Q3 - Q1
    up_lim = Q3 + 1.5 * IQR
    low_lim = Q1 - 1.5 * IQR
    df.loc[df[i] > up_lim,i] = up_lim
    df.loc[df[i] < low_lim,i] = low_lim

#Cat_list

for i in cat_list:
    plt.figure(figsize=(6,6))
    sns.countplot(x = i, data =df.loc[:,cat_list])
    plt.title(i)
    

#=====================

# K means clustering

from sklearn.cluster import KMeans
kmeans = KMeans()
kmeans.fit(df)

# run in a loop to know the no of clusters needed

l1 = []

for i in range(1,20):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(df)
    l1.append(kmeans.inertia_)
    
print(l1)

pd.DataFrame(range(1,20))        
pd.DataFrame(l1)
    
pd.concat([pd.DataFrame(range(1,20)),pd.DataFrame(l1)], axis=1)

plt.scatter(range(1,20),l1)
plt.show()    

plt.plot(range(1,20),l1)
plt.xlabel("k value")
plt.ylabel("wcss value")
plt.show()

# elbow plot
#pip install yellowbrick
from yellowbrick.cluster import KElbowVisualizer

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans,k=(1,20))
elbow.fit(df)
elbow.poof()
plt.show()

# optimum number of clusters 4 
# standarization
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X_std=ss.fit_transform(df)
X_std

kmeans=KMeans()
visualizer=KElbowVisualizer(kmeans,k=(1,20))
visualizer.fit(X_std)
visualizer.poof()
plt.show()

# Updated model

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4,n_init=20)
kmeans.fit(df)
cluster=kmeans.labels_
cluster

df["cluster_no"]=cluster
df.head()

df.cluster_no.value_counts()

#Hierarcihal Clustering
# single linkage method
import scipy.cluster.hierarchy as sch
plt.figure(figsize=(10,6))
plt.title("dendogram")
dendo = sch.dendrogram(sch.linkage(X_std,method='single'))

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage="single")
Y=cluster.fit_predict(X_std)

Y=pd.DataFrame(Y)
Y[0].value_counts()


# complete linkage method
plt.figure(figsize=(10,6))
plt.title("Crime rate dendogram")
dendo = sch.dendrogram(sch.linkage(X_std,method='single'))
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage="complete")
Y=cluster.fit_predict(X_std)

Y=pd.DataFrame(Y)
Y[0].value_counts()

# ward linkage method
plt.figure(figsize=(10,6))
plt.title("Crime rate dendogram")
dendo = sch.dendrogram(sch.linkage(X_std,method='single'))

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage="ward")
Y=cluster.fit_predict(X_std)

Y=pd.DataFrame(Y)
Y[0].value_counts()


#DBScan

from sklearn.cluster import DBSCAN

db = DBSCAN(eps=0.75,min_samples=3)
db.fit(X_std)
y=db.labels_
y=pd.DataFrame(y,columns=['cluster'])
y["cluster"].value_counts()

newdata =pd.concat([df,y],axis=1)

noisedata = newdata[newdata['cluster'] == -1]
print(noisedata)
finaldata = newdata[newdata['cluster'] == 0]
print(finaldata)









