import numpy as np
from sklearn.cluster import DBSCAN
from pipepine.core import ProcessingStep
from processing.radarproc import RpcFrame


class PointCloudPreprocessor(ProcessingStep):
    """
    A pre-processing step that prepares the 4D point cloud for clustering.
    Currently, this step applies a weight to the velocity dimension.
    """
    def __init__(self, velocity_weight: float):
        self.velocity_weight = velocity_weight

    def __call__(self, rpc_frame: RpcFrame) -> np.ndarray:
        """
        Args:
            rpc_frame (RpcFrame): The input radar frame data.

        Returns:
            np.ndarray: A (N, 4) numpy array with weighted velocity.
        """
        return np.column_stack((
            rpc_frame.x,
            rpc_frame.y,
            rpc_frame.z,
            np.multiply(rpc_frame.velocities, self.velocity_weight)
        ))


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


class DbscanClusterer(ProcessingStep):
    """
    A processing step that performs DBSCAN clustering on a point cloud.
    """
    def __init__(self, eps: float, min_samples: int):
        self.eps = eps
        self.min_samples = min_samples

    def __call__(self, point_cloud_4d: np.ndarray) -> FrameCluster:
        """
        Args:
            point_cloud_4d (np.ndarray): The (N, 4) input point cloud.

        Returns:
            FrameCluster: The result of the clustering.
        """
        model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        model.fit(point_cloud_4d)

        noise_mask = model.labels_ == -1
        cluster_mask = ~noise_mask
        
        return FrameCluster(point_cloud_4d, model.labels_, noise_mask, cluster_mask)
