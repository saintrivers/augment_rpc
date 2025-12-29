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

    def _elliptical_euclidean_scaling(self, points_xyz):
        w_x, w_y, w_z = 1.2, 0.6, 1.0

        scaled_points = np.copy(points_xyz)
        scaled_points[:, 0] *= w_x  # Scale X
        scaled_points[:, 1] *= w_y  # Scale Y
        scaled_points[:, 2] *= w_z  # Scale Z
        return scaled_points

    def __call__(self, point_cloud_4d: np.ndarray) -> FrameCluster:
        """
        Args:
            point_cloud_4d (np.ndarray): The (N, 4) input point cloud.

        Returns:
            FrameCluster: The result of the clustering.
        """
        scaled_point_cloud_4d = self._elliptical_euclidean_scaling(point_cloud_4d)
        model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        model.fit(scaled_point_cloud_4d)

        noise_mask = model.labels_ == -1
        cluster_mask = ~noise_mask
        
        return FrameCluster(point_cloud_4d, model.labels_, noise_mask, cluster_mask)

@dataclass
class ThreeDimensional:
    x: float
    y: float
    z: float

@dataclass
class Point(ThreeDimensional):
    pass
    
@dataclass
class BoxDimensions(ThreeDimensional):
    pass

@dataclass
class RadarObject:
    """
    Tracks a radar object with 3D centroid, Doppler velocity, and 3D bounding box size.
    """
    centroid: tuple[float, float, float]
    velocity: float
    size: BoxDimensions



class ClusterAnalyzer(ProcessingStep):
    def __init__(self, rpc_frame: RpcFrame):
        self.rpc_frame = rpc_frame
        
    def calculate_bounding_box(self, mask: np.ndarray) -> BoxDimensions:
        mean_x = self.rpc_frame.x[mask]
        mean_y = self.rpc_frame.y[mask]
        mean_z = self.rpc_frame.z[mask]
        
        dim_x = np.max(mean_x) - np.min(mean_x)
        dim_y = np.max(mean_y) - np.min(mean_y)
        dim_z = np.max(mean_z) - np.min(mean_z)

        return BoxDimensions(dim_x, dim_y, dim_z)

    def __call__(self, frame_cluster: FrameCluster):
        unique_labels = np.unique(frame_cluster.labels)
        moving_centroids = []
        
        for label in unique_labels:
            if label == -1:
                continue
            
            mask = frame_cluster.labels == label
            
            # --- 1. Quick Validations (Velocity & Count) ---
            # Velocity Check: Is it actually moving?
            raw_v = self.rpc_frame.velocities[mask]
            mean_v = np.mean(raw_v) 
            if np.abs(mean_v) < 1.0:
                continue

            # Density Check: A real car usually returns at least 3-4 points 
            # (unless it's extremely far away)
            num_points = np.sum(mask)
            # if num_points < 3: 
            #     continue

            # --- 2. Geometry Calculation ---
            shape = self.calculate_bounding_box(mask)
            # Sort dimensions so 'width' is always small, 'length' is always large
            width, length = sorted([shape.x, shape.y])
            
            # --- 3. The "Not-a-Wall" Filter ---
            # We remove the STRICT minimums. 
            # A bumper might be 1.8m wide and 0.2m deep. That is valid!
            
            # Max Width: A car/truck is rarely wider than 3m
            is_too_wide = width > 3.5 
            
            # Max Length: A car is ~5m, a truck ~10-15m. 
            # Anything > 15m is likely a guardrail or wall.
            is_too_long = length > 15.0 

            # Micro-Clutter: If it's 5cm x 5cm, it's likely noise (unless it has high velocity)
            is_micro_noise = (width < 0.2) and (length < 0.2)

            if is_too_wide or is_too_long or is_micro_noise:
                continue

            # --- 4. Success ---
            cluster_centroid = (
                np.mean(self.rpc_frame.x[mask]), 
                np.mean(self.rpc_frame.y[mask]), 
                np.mean(self.rpc_frame.z[mask])
            )
            
            moving_centroids.append(
                RadarObject(cluster_centroid, mean_v, shape)
            )
                    
        return moving_centroids, frame_cluster


class MdDbscanClusterer(ProcessingStep):
    """
    Implements the Multi-Dimensional DBSCAN from the paper.
    Stage 1: Cluster by Velocity.
    Stage 2: Cluster spatially within those velocity groups.
    """
    def __init__(self, spatial_eps: float, velocity_eps: float, min_samples: int):
        self.spatial_eps = spatial_eps      # e.g., 1.5 meters
        self.velocity_eps = velocity_eps    # e.g., 0.5 m/s (Doppler tolerance)
        self.min_samples = min_samples

    def __call__(self, point_cloud_4d: np.ndarray) -> FrameCluster:
        """
        point_cloud_4d columns: [x, y, z, velocity]
        """
        points_xyz = point_cloud_4d[:, :3]
        velocities = point_cloud_4d[:, 3].reshape(-1, 1) # Reshape for sklearn
        
        total_points = point_cloud_4d.shape[0]
        final_labels = np.full(total_points, -1, dtype=int) # Default to noise (-1)
        
        # --- STAGE 1: Velocity Clustering ---
        # We cluster ONLY on the velocity dimension first.
        # This groups "Static things", "Slow things", "Fast things".
        v_model = DBSCAN(eps=self.velocity_eps, min_samples=2)
        v_labels = v_model.fit_predict(velocities)
        
        # We need a counter to ensure cluster IDs don't overlap between groups
        global_cluster_id = 0

        # --- STAGE 2: Spatial Clustering (Per Velocity Group) ---
        unique_v_labels = np.unique(v_labels)
        
        for v_lab in unique_v_labels:
            if v_lab == -1:
                # If it's velocity noise, we leave it as global noise (-1)
                continue
                
            # Get the indices of points in this specific velocity group
            # e.g., "All points moving approx 20 m/s"
            indices = np.where(v_labels == v_lab)[0]
            subset_xyz = points_xyz[indices]
            
            # If there are not enough points to form a spatial cluster, skip
            if len(subset_xyz) < self.min_samples:
                continue

            # Run Spatial DBSCAN on this SUBSET
            s_model = DBSCAN(eps=self.spatial_eps, min_samples=self.min_samples)
            s_labels = s_model.fit_predict(subset_xyz)
            
            # Assign global IDs
            # We must shift the new labels so they don't collide with previous groups
            valid_mask = s_labels != -1
            
            if np.any(valid_mask):
                # Apply the spatial labels back to the main array
                # We add 'global_cluster_id' to make them unique across the whole frame
                final_labels[indices[valid_mask]] = s_labels[valid_mask] + global_cluster_id
                
                # Increment the global counter by the number of clusters found in this subset
                global_cluster_id += (np.max(s_labels) + 1)

        # Create masks for your visualizer
        noise_mask = final_labels == -1
        cluster_mask = ~noise_mask
        
        return FrameCluster(point_cloud_4d, final_labels, noise_mask, cluster_mask)