from sklearn.cluster import DBSCAN
import numpy as np

from processing.radarproc import RpcReplay


class FrameCluster:
    """
    Represents the result of clustering a single frame of a radar point cloud.

    Attributes:
        point_cloud (np.ndarray): The raw point cloud data used for clustering (N x 4 array).
        labels (np.ndarray): The cluster label for each point (-1 for noise).
        noise_mask (np.ndarray): A boolean mask where True indicates a noise point.
        cluster_mask (np.ndarray): A boolean mask where True indicates a point belonging to a cluster.
    """
    def __init__(self, point_cloud, labels, noise_mask, cluster_mask):
        self.point_cloud = point_cloud
        self.labels = labels
        self.noise_mask = noise_mask
        self.cluster_mask = cluster_mask


class RpcClustering:
    """
    A class that applies DBSCAN clustering to a RpcReplay object.

    This class acts as a factory for FrameCluster objects. It is initialized with
    the clustering hyperparameters and the full radar data replay. When indexed,
    it computes (and memos) the clustering result for the requested frame.
    """
    memo = {}

    def __init__(self, rpc_replay: RpcReplay, eps: float, min_samples: int, velocity_weight: float):
        """
        Initializes the RpcCluster with data and clustering parameters.

        Args:
            rpc_replay (RpcReplay): The pre-processed radar data sequence.
            eps (float): The maximum distance between two samples for one to be 
                        considered as in the neighborhood of the other (DBSCAN `eps`).
            min_samples (int): The number of samples in a neighborhood for a point
                                to be considered as a core point (DBSCAN `min_samples`).
            velocity_weight (float): A multiplier applied to the velocity dimension to
                                    adjust its influence on clustering distance.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.velocity_weight = velocity_weight
        self.rpc_replay = rpc_replay

    def __getitem__(self, idx: int) -> FrameCluster:
        """
        Computes or retrieves the clustering result for a specific frame.

        Args:
            idx (int): The index of the frame to cluster.

        Returns:
            FrameCluster: An object containing the clustering results for the frame.
        """
        if idx in self.memo:
            return self.memo[idx]

        rpc_frame = self.rpc_replay[idx]

        point_cloud_4d = np.column_stack((rpc_frame.x, rpc_frame.y, rpc_frame.z, np.multiply(rpc_frame.velocities, self.velocity_weight)))
        model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        model.fit(point_cloud_4d)

        noise_mask = model.labels_ == -1
        cluster_mask = ~noise_mask
        
        cluster = FrameCluster(point_cloud_4d, model.labels_, noise_mask, cluster_mask)
        self.memo[idx] = cluster
        return cluster
