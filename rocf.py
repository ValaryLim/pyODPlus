# import packages
import numpy as np
from scipy.spatial.distance import cityblock, euclidean # distance metrics
from heapq import heappush, heappop # priority queue
from queue import Queue # queue

class ROCF():
    def __init__(self, distance_metric="euclidean", k=3, threshold=0.1):
        '''
        Parameters
        ----------
        distance_metric : str in ("manhattan", "euclidean"), optional 
            (default="euclidean")
            The distance metric to use to compute k nearest neighbours.
        
        k : int, optional (default=3)
            k number of nearest neigbours used to form MUtual Neighbour Graph.
        
        threshold : float, optional (default=0.1)
            Threshold set for any cluster to be considered an outlier. 
            Each cluster has an ROCF value.
            If max({ROCF}) < threshold, no cluster is considered as outlier.
            Else, all clusters with smaller size than cluster with max ROCF are
            tagged as outliers. 
        '''
        # checks for input validity
        if distance_metric not in ["euclidean", "manhattan"]:
            raise ValueError("Invalid distance_metric input. Only accepts 'euclidean' or 'manhattan'.")
        
        try:
            if int(k) != k:
                raise ValueError("Invalid k input. k should be an integer")
        except: 
            raise ValueError("Invalid k input. k should be an integer")
        
        try: 
            if float(threshold) != threshold or threshold < 0 or threshold > 1:
                raise ValueError("Invalid threshold input. threshold should be a float between 0 and 1")
        except:
            raise ValueError("Invalid threshold input. threshold should be a float between 0 and 1")


        # initialise input attributes
        self.distance_metric = distance_metric
        self.k = int(k)
        self.threshold = threshold
        
        # define computed attributes
        self.outliers = None
        self.transition_levels = None
        self.rocfs = None
        self.cluster_labels = None
        self.cluster_groups = None
        self.k_nearest_neighbours = None

    def fit(self, X):
        '''
        Runs ROCF algorithm to detect outliers.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        Returns
        ----------
        self : object
            Fitted estimator.
        '''
        # retrieve k nearest neighbours
        k_nearest_neighbours = self._retrieve_k_nearest_neighbours(X)

        # cluster datasets using mutual neighbour graph
        cluster_labels, cluster_groups = self._retrieve_clusters_mung(X, k_nearest_neighbours)

        # compute outliers
        outliers, transition_levels, rocfs = self._compute_outliers_rocf(X, cluster_groups)

        # update self object
        self.k_nearest_neighbours = k_nearest_neighbours
        self.cluster_labels = cluster_labels
        self.cluster_groups = cluster_groups
        self.outliers = outliers
        self.transition_levels = transition_levels
        self.rocfs = rocfs

        return self
 
    def _compute_distance(self, v1, v2):
        '''
        Computes distance between two data points
        
        Parameters
        ----------
        v1 : numpy array of shape (n_features,)
            The first data point.
        
        v2 : numpy array of shape (n_features,)
            The second data point

        Returns
        ----------
        distance : float
            Distance between two input data points.
        '''
        distance_metric = self.get_distance_metric()
        if distance_metric == "euclidean":
            return euclidean(v1, v2)
        else:
            return cityblock(v1, v2)
    
    def _retrieve_k_nearest_neighbours(self, X):
        '''
        Retrieves k nearest neighbours
        
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        Returns
        ----------
        k_nearest_neighbours: numpy array of shape (n_samples, k_clusters)
        '''
        k_nearest_neighbours = []
        k = self.get_k()

        # iterate through each element in X and find the k nearest neighbours
        for i in range(len(X)):
            # create heap to store k nearest neighbours
            neighbours_i = []
            
            # iterate through all other points
            for j in range(len(X)):
                if i == j:
                    continue # do not count itself
                
                # compute distance between i and j
                current_dist = self._compute_distance(X[i], X[j])
                
                if len(neighbours_i) < k:
                    # insufficient neighbours, add into neighbour heap
                    heappush(neighbours_i, (-current_dist, j))
                
                if len(neighbours_i) == k: 
                    # retrieve largest distance neighbour
                    largest_neighbour = heappop(neighbours_i)
                    
                    if current_dist < -largest_neighbour[0]:
                        # distance between i and j is smaller than largest neighbour
                        # update neighbours to include current element
                        heappush(neighbours_i, (-current_dist, j))
                    else:
                        # replace largest element back in neighbours heap
                        heappush(neighbours_i, largest_neighbour)
                        
            # extract the neighbours
            neighbours_i = set([x[1] for x in neighbours_i])
                    
            k_nearest_neighbours.append(neighbours_i)

        return k_nearest_neighbours

    def _retrieve_clusters_mung(self, X, k_nearest_neighbours):
        '''
        Retrieves clusters using the MUtual Neighbour Graph (MUNG) algorithm
        
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        k_nearest_neighbours : numpy array of shape (n_samples, k_clusters)
            The k nearest neighbours of each sample.
        
        Returns
        ----------
        cluster_labels : numpy array of shape (n_samples,)
            Array indicating the cluster that each sample is classified as
        
        cluster_groups : numpy array of shape (n_clusters, 2)
            Each sample row contains (size of cluster, samples in cluster) 
        '''
        # define visited
        visited = [False] * len(X)

        # define clusters for each point (initialise to -1)
        cluster_labels = [-1] * len(X)

        # define cluster groups
        cluster_groups = []

        label = 0 # counter for cluster label

        # iterate through each element 
        for i in range(len(X)):
            if visited[i]: # ignore if element is already visited
                continue
        
            # define queue to store mutual neighbours of cluster
            mutual_neighbours = Queue()
            mutual_neighbours.put(i)

            # define cluster group
            current_cluster = set()

            # while there still exists mutual neighbours
            while not mutual_neighbours.empty():
                # retrieve next mutual neighbour
                v = mutual_neighbours.get()

                # mark as visited, label cluster and add to group
                visited[v] = True
                cluster_labels[v] = label
                current_cluster.add(v)

                # find all unvisited mutual neighbours of v
                for v_neighbours in k_nearest_neighbours[v]:
                    if (v in k_nearest_neighbours[v_neighbours]) and (not visited[v_neighbours]):
                        # if v is a mutual neighbour and is not visited
                        mutual_neighbours.put(v_neighbours)
            
            # update cluster group
            cluster_groups.append([len(current_cluster), current_cluster])

            label += 1 # increment label

        return np.array(cluster_labels), np.array(cluster_groups)
    
    def _compute_outliers_rocf(self, X, cluster_groups):
        '''
        Computes outliers using the ROCF method
        
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        cluster_groups : numpy array of shape (n_clusters, 2)
            Each sample row contains (size of cluster, samples in cluster) 
        
        Returns
        ----------
        outliers : numpy array of shape (n_samples,)
            Array indicating if sample is outlier (1) or normal (0)
        
        transition_levels : numpy array of shape (n_clusters - 1,)
            Transition level of each cluster

        rocfs : numpy array of shape (n_clusters - 1,)
            ROCF of each cluster
        '''
        # define outliers
        outliers = [0] * len(X) # 0 marks normal data point

        # filter out all cluster groups smaller than k
        k = self.get_k()

        # points in clusters of size < k are outleirs
        for cg in cluster_groups:
            if self._get_cluster_size(cg) < k:
                # cluster is an outlier cluster
                cg_points = self._get_cluster_points(cg)
                for v in cg_points:
                    # tag all points in cluster as outliers
                    outliers[v] = 1 
        
        # sort clusters
        cluster_groups_sorted = sorted(cluster_groups, key=lambda x: self._get_cluster_size(x), reverse=False)

        # compute transition levels and rocf
        transition_levels = []
        rocfs = [] 
        # iterate through cluster groups
        for i in range(len(cluster_groups_sorted) - 1):
            c1_size = self._get_cluster_size(cluster_groups_sorted[i])
            c2_size = self._get_cluster_size(cluster_groups_sorted[i+1])
            tl = c2_size / c1_size
            rocf = 1 - np.exp(-tl / c1_size)

            # update transition levels and rocfs
            transition_levels.append(tl)
            rocfs.append(rocf)
        
        # retrieve maximum rocfs
        max_rocf = max(rocfs)
        max_rocf_index = max(ind for ind, value in enumerate(rocfs) if value == max_rocf)

        # identify outliers from maximum rocfs
        threshold = self.get_threshold()
        if max_rocf > threshold: # if greater than threshold, some clusters are outliers
            for i in range(max_rocf_index): 
                for v in self._get_cluster_points(cluster_groups_sorted[i]):
                    outliers[v] = 1 # tag points in outlier clusters
        
        return np.array(outliers), np.array(transition_levels), np.array(rocfs)

    def get_outliers(self):
        return self.outliers
    
    def get_k_nearest_neighbours(self):
        return self.k_nearest_neighbours
    
    def get_cluster_labels(self):
        return self.cluster_labels
    
    def get_cluster_groups(self):
        return self.cluster_groups

    def get_outlier_rate(self):
        return sum(self.outliers) / len(self.outliers)
    
    def get_transition_levels(self):
        return self.transition_levels
    
    def get_rocfs(self):
        return self.rocfs
    
    def _get_cluster_size(self, cluster):
        return cluster[0]
    
    def _get_cluster_points(self, cluster):
        return cluster[1]
    
    def get_k(self):
        return self.k
    
    def get_threshold(self):
        return self.threshold
    
    def get_distance_metric(self):
        return self.distance_metric
