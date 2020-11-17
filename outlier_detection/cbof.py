import numpy as np
from scipy.spatial.distance import cityblock, euclidean # distance metrics
from heapq import heappush, heappop # priority queue
from queue import Queue # queue

class CBOF():
    def __init__(self, distance_metric="euclidean", k=5, lofub=2.0, pct=0.5, \
        contamination=0.1):
        '''
        Parameters
        ----------
        distance_metric : str in ("manhattan", "euclidean"), optional 
            (default="euclidean")
            The distance metric to use to compute k nearest neighbours.
        
        k : int, optional (default=5)
            k number of nearest neigbours to compute Local Outlier Factor 
            and Local Reachability Density.
        
        lofub : float, optional (default=2.0)
            Threshold set for any point to be considered a core point in a 
            cluster.
            If LOF(p) <= lofub, p is a core point. 
        
        pct : float, optional (default=0.5)
            Value in range (0, 1]
            Percentage to consider if points are local density reachable 
            from one another
        
        contamination : float, optional (default=0.1)
            Percentage of outliers in dataset
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
            if float(pct) != pct or pct <= 0 or pct > 1:
                raise ValueError("Invalid pct input. pct should be a float between 0 and 1")
        except:
            raise ValueError("Invalid threshold input. threshold should be a float between 0 and 1")
            
        try: 
            if float(contamination) != contamination or contamination < 0 or contamination > 1:
                raise ValueError("Invalid contamination input. contamination should be a float between 0 and 1")
        except:
            raise ValueError("Invalid contamination input. contamination should be a float between 0 and 1")

        try: 
            if float(lofub) != lofub or lofub < 0:
                raise ValueError("Invalid lofub input. lofub should be a float greater than 0")
        except:
            raise ValueError("Invalid lofub input. lofub should be a float greater than 0")
            
        # initialise input attributes
        self.distance_metric = distance_metric
        self.k = int(k)
        self.lofub = float(lofub)
        self.pct = float(pct)
        self.contamination = float(contamination)
        self.alpha = 1 - float(contamination)
        
        # define computed attributes
        self.k_nearest_neighbours = None
        self.cluster_labels = None
        self.cluster_groups = None
        self.outliers = None
        self.local_reachability_densities = None
        self.local_outlier_factors = None
        self.core_points = None
        self.ldr_points = None
    
    def fit(self, X):
        '''
        Runs CBOF algorithm to detect outliers.

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

        # compute reachability distance
        reachability_distances = self._minpts_reachability_distance(k_nearest_neighbours)
        
        # compute local reachability densities
        local_reachability_densities = self._compute_lrd(reachability_distances)

        # compute local outlier factors
        local_outlier_factors = self._compute_lof(local_reachability_densities, k_nearest_neighbours)

        # retrieve core and ldr points
        core_points = self._retrieve_core_points(local_outlier_factors)
        ldr_points = self._retrieve_ldr_points(local_reachability_densities, k_nearest_neighbours)

        # retrieve cluster labels and groups
        # noise = -1
        cluster_labels, cluster_groups = self._retrieve_clusters(core_points, ldr_points)

        # retrieve outlier labels
        outliers = self._label_outliers(cluster_labels, cluster_groups)

        # update self object
        self.k_nearest_neighbours = k_nearest_neighbours
        self.cluster_labels = cluster_labels
        self.cluster_groups = cluster_groups
        self.outliers = outliers
        self.local_reachability_densities = local_reachability_densities
        self.local_outlier_factors = local_outlier_factors
        self.core_points = core_points
        self.ldr_points = ldr_points

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
            neighbours_i = [[-x[0], x[1]] for x in neighbours_i]
            
            k_nearest_neighbours.append(neighbours_i)

        return k_nearest_neighbours

    def _minpts_reachability_distance(self, k_nearest_neighbours):
        reachability_distances = []
        for p_ind in range(len(k_nearest_neighbours)):
            rd = []
            for j in range(len(k_nearest_neighbours[0])):
                # retrieve k distance
                q_ind = k_nearest_neighbours[p_ind][j][1]
                k_distance_q = max([x[0] for x in k_nearest_neighbours[q_ind]])
                
                # retrieve distance from p to q
                dist_p_q = k_nearest_neighbours[p_ind][j][0]
                
                rd.append(max(k_distance_q, dist_p_q))

            reachability_distances.append(rd)
        
        return reachability_distances
    
    def _compute_lrd(self, reachability_distances):
        local_reachability_densities = []
        for p_ind in range(len(reachability_distances)):
            # compute average reachability distance
            avg_rd = np.mean(reachability_distances[p_ind])
            # lrd = inverse average reachability distance
            local_reachability_densities.append(1 / avg_rd)
        return local_reachability_densities

    def _compute_lof(self, local_reachability_densities, k_nearest_neighbours):
        local_outlier_factors = []
        for p_ind in range(len(k_nearest_neighbours)):
            total = 0
            neighbours = [x[1] for x in k_nearest_neighbours[p_ind]]
            for n_ind in neighbours:
                total += local_reachability_densities[n_ind] / local_reachability_densities[p_ind]
            lof = total / len(neighbours)
            
            local_outlier_factors.append(lof)
        return local_outlier_factors

    def _retrieve_core_points(self, local_outlier_factors):
        lofub = self.get_lofub()
        return [True if x <= lofub else False for x in local_outlier_factors]
    
    def _retrieve_ldr_points(self, local_reachability_densities, k_nearest_neighbours):
        pct = self.get_pct()
        # create set of ldr points
        ldr_points = [set() for i in range(len(k_nearest_neighbours))]
        
        for q_ind in range(len(k_nearest_neighbours)):
            q_neighbours = k_nearest_neighbours[q_ind]
            for p_ind in range(len(q_neighbours)):
                # retrieve lrd of p and q
                lrd_q = local_reachability_densities[q_ind]
                lrd_p = local_reachability_densities[p_ind]
                # condition
                if (lrd_q / (1+pct) < lrd_p) and (lrd_q * (1+pct) > lrd_p):
                    # point p is local density reachable from q
                    ldr_points[q_ind].add(q_neighbours[p_ind][1])
                    ldr_points[q_neighbours[p_ind][1]].add(q_ind)
        
        return ldr_points

    def _retrieve_clusters(self, core_points, ldr_points):
        '''
        noise is classified as -1
        '''
        # define visited
        visited = [False] * len(ldr_points)
        
        # define clusters for each point
        cluster_labels = [-1] * len(ldr_points)
        
        # define cluster groups
        cluster_groups = []
        
        label = 0 # counter for cluster label
        
        # iterate through each element in ldr_points
        for i in range(len(ldr_points)): 
            if visited[i] or not core_points[i]:
                continue # skip, already clustered OR not core point in cluster
            
            # define queue to store neighbours of cluster
            neighbours = Queue()
            neighbours.put(i)
            visited[i] = True
            cluster_labels[i] = label
            
            # define cluster group
            current_cluster = set()
            current_cluster.add(i)
            
            while not neighbours.empty(): # while there are still unexplored neighbours in cluster
                # retrieve next neighbour
                v = neighbours.get()
                
                # find all unvisited neighbours of v
                for v_reachable in ldr_points[v]:
                    if (not visited[v_reachable]):
                        # update labels
                        visited[v_reachable] = True
                        cluster_labels[v_reachable] = label
                        current_cluster.add(v_reachable)
                        # put neighbour
                        neighbours.put(v_reachable)
                
            # update cluster group
            cluster_groups.append([len(current_cluster), current_cluster])
            
            label += 1 # increment label
        
        return np.array(cluster_labels), np.array(cluster_groups)

    def _label_outliers(self, cluster_labels, cluster_groups):
        alpha = self.get_alpha()

        # order clusters in decreasing size
        cluster_groups_sorted = sorted(cluster_groups, key=lambda x: x[0], reverse=True)
        
        threshold = len(cluster_labels) * alpha
        
        current_total = 0
        
        # compute upper bound for outlier cluster
        outlier_cluster_upper_bound = 0
        for i in range(len(cluster_groups_sorted)):
            current_total += cluster_groups_sorted[i][0] # add to total number of objects
            if current_total >= threshold:
                outlier_cluster_upper_bound = cluster_groups_sorted[i][0]
                break
        
        # label outliers in cluster labels
        outlier_labels = list(cluster_labels)[:] # make copy
        for i in range(len(cluster_groups_sorted)):
            if cluster_groups_sorted[i][0] > outlier_cluster_upper_bound:
                continue # not outlier
            for j in cluster_groups_sorted[i][1]:
                outlier_labels[j] = -1
        
        return np.array([1 if x==-1 else 0 for x in outlier_labels])

    def get_k(self):
        return self.k
    
    def get_lofub(self):
        return self.lofub
    
    def get_pct(self):
        return self.pct

    def get_alpha(self):
        return self.alpha
    
    def get_contamination(self):
        return self.contamination

    def get_outliers(self):
        return self.outliers
    
    def get_k_nearest_neighbours(self):
        return self.k_nearest_neighbours
    
    def get_cluster_labels(self):
        return self.cluster_labels
    
    def get_cluster_groups(self):
        return self.cluster_groups

    def get_distance_metric(self):
        return self.distance_metric