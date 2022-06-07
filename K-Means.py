# from matplotlib.pyplot import axes
import numpy as np
import pandas as pd
# from sklearn import cluster
""" 
!RM'd tabnine keyBind 
{
  "key": "tab",
  "command": "tabnine.accept-inline-suggestion",
  "when": "tabnine.in-inline-suggestions && tabnine.inline-suggestion:enabled || tabnine.in-inline-suggestions && tabnine.snippet-suggestion:enabled"
}
"""

def randomClusterInit(X, rowNum, clusterNum):
    return X[np.random.choice(rowNum, size=clusterNum, replace=False)]


""" 
p = np.random.randn(4,4)
print(p)
c = np.random.randint(10,90,(4,5))
print(c)
print(c.shape)
print(c.shape[0])

print(len(c)) ---> rowcount
print(len(c[0])) ---> colCount
print(randomClusterInit(c, c.shape[0], 4)) 
c = np.random.randint(10,90,(4,5))
"""




class KMeans:

    def __init__(self, initMeth, clusterNum, tolerance=0.01, epochs=100, runs=1):
        self.clusterNum = clusterNum
        self.tolerance = tolerance
        self.epochs = epochs
        self.clusterMeans = np.zeros(clusterNum)
        self.initMeth = initMeth
        self.runs = runs

    def fit(self, X):
        rowNum, colNum = X.shape
        X_vals = self.__get_values(X)
        X_labels = np.zeros(rowNum)

        cost = np.zeros(self.runs)

        allClusters = []
        for i in range(self.runs):
            clusterMeans = self.__initialize_means(X_vals, rowNum)

            for _ in range(self.epochs):
                prevMean = np.copy(clusterMeans)

                dist = self.__computeDistance(X_vals, clusterMeans, rowNum)

                X_labels = self.__label_examples(dist)
                clusterMeans = self.__computeMeans(X_vals, X_labels, colNum)

                convergenceCheck = np.abs(clusterMeans - prevMean)<self.tolerance
                if np.all(convergenceCheck):
                    break
            X_valsWithLabels = np.append(X_vals, X_labels[:, np.newaxis], axis=1) 
            
            allClusters.append((clusterMeans, X_valsWithLabels))
            cost[i] = self.__computeCost(X_labels, X_labels, clusterMeans)

            BestClusterIndex = cost.argmin()
            self.cost_ = cost[BestClusterIndex]

            
            return allClusters[BestClusterIndex]


    def __computeMeans(self,X, labels, colNum):
        clusterMeans = np.zeros((self.clusterNum, colNum))
        # rmd ,_ from loop iter 
        for clusterMeanIndex in enumerate(clusterMeans): 
            clusterElements =  X[labels == clusterMeanIndex]
            
            if len(clusterElements):
                clusterMeans[clusterMeanIndex, :] = clusterElements.mean(axis=0)
        return clusterMeans 

    def __get_values(self, X):
        if isinstance(X, np.ndarray):
            return X
        return np.array(X) 

    def __computeCost(self, X, labels, clusterMeans): 
        cost =  0 
        for clusterMeanIndex, clusterMean in enumerate(clusterMeans): 
            clusterElements = X[labels == clusterMeanIndex]
            # cost += np.linalg.norm(clusterElements-clusterMean, axis=1).sum()
        return cost 
    
    def __initialize_means(self, X, rowNum):
        # selfInit = randomClusterInit(X, rowNum, self.clusterNum)
        # print(selfInit) 
        return randomClusterInit(X, rowNum, self.clusterNum) 
    
    def __computeDistance(self, X, clusterMeans, rowNum):
        distances = np.zeros((rowNum,self.clusterNum))

        for clusterMeanIndex, clusterMean, in enumerate(clusterMeans):
            distances[:, clusterMeanIndex] = np.linalg.norm(X - clusterMean, axis=1)
        return distances
    
    def __label_examples(self, distances): 
        return distances.argmin(axis=1)
