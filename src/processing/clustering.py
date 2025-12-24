from dataclasses import dataclass
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

@dataclass
class FrameCluster:
    """
    Represents the result of clustering a single frame of a radar point cloud.

    Attributes:
        point_cloud (np.ndarray): The raw point cloud data used for clustering (N x 4 array).
        labels (np.ndarray): The cluster label for each point (-1 for noise).
        noise_mask (np.ndarray): A boolean mask where True indicates a noise point.
        cluster_mask (np.ndarray): A boolean mask where True indicates a point belonging to a cluster.
    """
    point_cloud: np.ndarray
    labels: np.ndarray
    noise_mask: np.ndarray
    cluster_mask: np.ndarray


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


@dataclass
class RadarObject:
    """
    Tracks a radar object with 3D centroid, Doppler velocity, and 3D bounding box size.
    """
    centroid: tuple[float, float, float]
    velocity: float
    size: tuple[float, float, float]


class ClusterAnalyzer(ProcessingStep):
    def __init__(self, rpc_frame: RpcFrame):
        self.rpc_frame = rpc_frame

    def __call__(self, frame_cluster: FrameCluster):
        # frame_cluster.labels has the IDs (e.g., 0, 0, 1, -1, 0...)
        # self.rpc_frame.velocities has the RAW velocities
        
        unique_labels = np.unique(frame_cluster.labels)
        
        moving_centroids = []
        
        for label in unique_labels:
            if label == -1:
                continue
            mask = frame_cluster.labels == label
            raw_v = self.rpc_frame.velocities[mask]
            mean_v = np.mean(raw_v) 

            mean_x = self.rpc_frame.x[mask]
            mean_y = self.rpc_frame.y[mask]
            mean_z = self.rpc_frame.z[mask]
            cluster_centroid = np.mean(mean_x), np.mean(mean_y), np.mean(mean_z)
            
            shape = (
                np.max(mean_x) - np.min(mean_x), 
                np.max(mean_y) - np.min(mean_y), 
                np.max(mean_z) - np.min(mean_z)
                )
            
            if np.abs(mean_v) >= 0.5:
                moving_centroids.append(
                    RadarObject(cluster_centroid, mean_v, shape)
                )
        
        return moving_centroids, frame_cluster