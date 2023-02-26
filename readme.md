# Density-Based Spatial Clustering in Rain and Temperature

### Problem Statement:

DBSCAN Density-Based Spatial Clustering is an algorithm for clustering data that is commonly
used in machine learning and data mining. It is based on the idea of density-reachability, which
means that it can identify clusters of points that are closely packed together and separate them
from points that are more spread out.
We can use DBSCAN to cluster weather data that includes temperature and rain measurements.
We use it to identify patterns in the relationship between temperature and rain by grouping
together data points that have similar temperature and rain values. This could be useful for
understanding how temperature and rain vary in different parts of our country at different times
of the year.
However, keep in mind that DBSCAN is just one tool among many that can be used to analyze
weather data, and there may be other methods that are better suited to your specific research
questions.
Finding density-based clustering based on connected regions with high density.

### System Requirements:

1. Processor: Intel Core I5 Processor Or Equivalent. (Preferable )
2. RAM: 2GB RAM (4GB preferable)
3. Operating System: Windows 7 to 10
4. IDE Used: Jupyter Notebook and Google Colab

### System Design:

Machine learning algorithm that is used in this project is an unsupervised learning method called
Density-Based clustering. In this clustering algorithm, it takes two values as inputs. One is
epsilon ε which is the radius of the density circle, meaning the maximum distance between two
points in the same cluster and another is the minimum points. The surroundings with a radius ε
of a given point are known as the ε neighbourhood of the point. If the ε neighbourhood of the
point comprises at least a minimum number (min points), then it is called a core point. The data
points in the region are separated by two clusters. High-density points and low-density points.
Low point density is considered as noise.
In this project, we have taken a dataset of Bangladesh’s weather with 4 features such as year,
month, temperature, and rain. Then we fitted the dataset into the model.

![img](https://user-images.githubusercontent.com/38730778/221421984-77874604-8aeb-42e3-8aeb-d6a6326b9132.JPG)

### Implementation:

```
clusters = fit_predict(X,0.1,4)
```

Here the program passes the input values to the fit_predict function.

```
def fit_predict(X,eps,minPts):
  clusters = [0]*X.shape[0]

  dbscan(X,clusters,eps,minPts,metric=distance.euclidean)

  return clusters
```

where the dataset, clusters, eps, minimum point, and metric value are being passed to the dbscan
function, and the value of the clusters is being updated

```
def dbscan(X,clusters,eps,minPts,metric=distance.euclidean):
  currentPoint=0

  for i in range(0,X.shape[0]):
    if clusters[i] != 0:
      continue

    neighbors = neighborsGen(X,i,eps,metric)  # check number of neighbors

    if len(neighbors) < minPts:       # check core point
      clusters[i] = -1
    else:
      currentPoint += 1
      expand(X,clusters,i,neighbors,currentPoint,eps,minPts,metric)    #expand chain with core points and add clusters

  return clusters
```

### Testing Results:

![res](https://user-images.githubusercontent.com/38730778/221422144-4e4d523d-a704-44be-a7e1-7161b2da5d77.JPG)


#anik