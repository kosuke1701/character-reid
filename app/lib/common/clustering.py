from sklearn.cluster import AgglomerativeClustering

class AgglomerativeCluster(object):
    def __init__(self):
        self.model = None
    
    def fit(self, distance_matrix):
        """
        Parameters:
            distance_matrix (numpy.ndarray) - 2D distance matrix of shape (num_image x num_image).
        """
        self.model = AgglomerativeClustering(
            distance_threshold=0, n_clusters=None,
            affinity="precomputed", linkage="average"
        )
        self.model.fit(distance_matrix)

        self.n_img = distance_matrix.shape[0]
    
    def get_range(self):
        """
        Return: (min_level, max_level)
        """
        if self.model is None:
            raise Exception("Cluster model is not constructed.")

        depth = self.model.children_.shape[0]

        return (0, depth)
    
    def get_clusters(self, level):
        """
        Return: clusters
            clusters - A list of tuples. Each tuple contains image indices in a cluster.
        """
        # Merge clusters `level` times
        current_clusters = {i_img: [i_img] for i_img in range(self.n_img)}
        for i_step, (cls1, cls2) in enumerate(self.model.children_):
            if i_step >= level:
                break
            
            current_clusters[self.n_img + i_step] = \
                current_clusters[cls1] + current_clusters[cls2]
            del current_clusters[cls1], current_clusters[cls2]
        
        # Sort clusters.
        clusters = [tuple(sorted(clst)) for clst in current_clusters.values()]
        clusters = sorted(clusters)

        return clusters

if __name__=="__main__":
    import numpy as np
    clst = AgglomerativeCluster()

    dist_mat = np.array([[0.0, 0.1,0.3],[0.1, 0.0, 0.4],[0.3,0.4,0.0]])
    clst.fit(dist_mat)
    for i in range(clst.get_range()[1]+1):
        print(clst.get_clusters(i))
