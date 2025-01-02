import numpy as np
import random as rd

class KMeans:
    
    def __init__(self,n_clusters=2,max_iters=100):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.centroid = None

    def fit_predict(self,x):
        rand_idx = rd.sample(range(0,x.shape[0]),self.n_clusters)
        self.centroid = x[rand_idx]

        for i in range(self.max_iters):
            cluster_group = self.assign_clusters(x)
            old_centroid = self.centroid
            self.centroid = self.move_centroid(x,cluster_group)
            if (old_centroid==self.centroid).all():
                break

        return cluster_group

    def assign_clusters(self,x):
        cluster_group = []
        for r in x:
            distance = []
            for c in self.centroid:
                distance.append(np.sqrt(np.dot(r-c,r-c)))
            min_dist = min(distance)
            min_idx = distance.index(min_dist)
            cluster_group.append(min_idx)
        return np.array(cluster_group)

    def move_centroid(self,x,cluster_group):
        new_centroid = []
        cluster_type = np.unique(cluster_group)
        for type in cluster_type:
            new_centroid.append(x[cluster_group==type].mean(axis=0))
        return  np.array(new_centroid)