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


class RpcClusterFactory:
    """
    A class that applies DBSCAN clustering to a RpcReplay object.

    This class acts as a factory for FrameCluster objects. It is initialized with
    the full radar data replay. When called, it computes (and memos) the
    clustering result for the requested frame and hyperparameters.
    """
    def __init__(self, rpc_replay: RpcReplay):
        """
        Initializes the RpcClusterFactory with data.

        Args:
            rpc_replay (RpcReplay): The pre-processed radar data sequence.
        """
        self.rpc_replay = rpc_replay
        self._memo = {}

    def get_cluster(self, idx: int, eps: float, min_samples: int, velocity_weight: float) -> FrameCluster:
        """
        Computes or retrieves the clustering result for a specific frame.

        Args:
            idx (int): The index of the frame to cluster.
            eps (float): The DBSCAN `eps` parameter.
            min_samples (int): The DBSCAN `min_samples` parameter.
            velocity_weight (float): The weight for the velocity dimension.

        Returns:
            FrameCluster: An object containing the clustering results for the frame.
        """
        # Create a key that uniquely identifies this clustering request
        cache_key = (idx, eps, min_samples, velocity_weight)
        if cache_key in self._memo:
            return self._memo[cache_key]

        rpc_frame = self.rpc_replay[idx]

        point_cloud_4d = np.column_stack((
            np.multiply(rpc_frame.x, 1), 
            np.multiply(rpc_frame.y, 1), 
            np.multiply(rpc_frame.z, 1), 
            np.multiply(rpc_frame.velocities, velocity_weight)
        ))
        model = DBSCAN(eps=eps, min_samples=min_samples)
        model.fit(point_cloud_4d)

        noise_mask = model.labels_ == -1
        cluster_mask = ~noise_mask
        
        cluster = FrameCluster(point_cloud_4d, model.labels_, noise_mask, cluster_mask)
        self._memo[cache_key] = cluster
        return cluster
