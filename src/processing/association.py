import numpy as np
from scipy.optimize import linear_sum_assignment
from pipepine.factory import RpcProcessFactory
from processing.clustering import FrameCluster, RadarObject
import copy

from processing.radarproc import estimate_ego_velocity


class HungarianMatcher:
    def __init__(self, max_distance: float = 2.0):
        """
        Docstring for __init__
        
        :param self: Description
        :param max_distance: Sets the distance in meters (measured by the CARLA simulator) to be used for matching.
        :type max_distance: float
        """
        self.max_distance = max_distance
        
    def __call__(self, detections_prev: list[RadarObject], detections_curr: list[RadarObject]) -> dict[int, int]:
        num_prev = len(detections_prev)
        num_curr = len(detections_curr)

        if num_prev == 0 or num_curr == 0:
            return {}

        # 1. Create Cost Matrix (Vectorized for speed)
        # Extract centroids into arrays for fast broadcasting
        prev_cents = np.array([obj.centroid for obj in detections_prev])
        curr_cents = np.array([obj.centroid for obj in detections_curr])
        
        # Force both arrays to use only the first 2 dimensions (x, y)
        prev_cents = prev_cents[:, :2]
        curr_cents = curr_cents[:, :2]
        
        # Calculate Euclidean distance between all pairs
        # Result is (N, M) matrix where entry [i, j] is dist between prev[i] and curr[j]
        # Using linalg.norm on the difference
        diff = prev_cents[:, np.newaxis, :] - curr_cents[np.newaxis, :, :]
        cost_matrix = np.linalg.norm(diff, axis=2)

        # 2. Run Hungarian Algorithm (Munkres)
        # Note: We do NOT set np.inf here. We let it find the best geometric matches first.
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # 3. Filter Invalid Matches (Gating)
        associations = {}
        for r, c in zip(row_ind, col_ind):
            dist = cost_matrix[r, c]
            if dist < self.max_distance:
                # Only accept the match if it's within our physical threshold
                associations[c] = r 
            else:
                # If the "best" match is 50 meters away, it's not a match.
                # We ignore this, effectively treating it as a "Lost Track" for r 
                # and a "New Object" for c.
                pass
            
        return associations


def _get_compensation_values(timestamp_curr, timestamp_prev, point_cloud, ego_gyro):
    dt = timestamp_curr - timestamp_prev
    if dt <= 0:
        return 0.0, 0.0, 0.0

    # If point_cloud is a list of arrays, stack them into one array
    x = point_cloud[:,0]
    y = point_cloud[:,1]
    z = point_cloud[:,2]
    v = point_cloud[:,3]
    v_ego = estimate_ego_velocity((x,y,z,v), ego_gyro)
    
    # 2. Calculate Translation in the CAR'S frame
    # (How many meters did the car move forward/left?)
    dx_ego = v_ego[0] * dt
    dy_ego = v_ego[1] * dt

    # 3. Get Rotation angle (Yaw change)
    # gyro is usually [roll_rate, pitch_rate, yaw_rate]
    # We only care about Z (yaw) for 2D matching
    d_theta = ego_gyro[2] * dt
    
    return dx_ego, dy_ego, d_theta


def _compensate_motion(objects, dx, dy, dtheta):
    # Rotation matrix for the change in heading
    c, s = np.cos(dtheta), np.sin(dtheta)
    R = np.array(((c, s), (-s, c))) 

    for obj in objects:
        # 1. Translate (Shift the world back by how much we moved)
        # Note: Direction depends on your coordinate convention (usually subtract ego motion)
        shifted_x = obj.centroid[0] - dx
        shifted_y = obj.centroid[1] - dy
        
        # 2. Rotate (Adjust for car turning)
        # Apply rotation to the shifted point
        obj.centroid = R.dot(np.array([shifted_x, shifted_y]))


def hungarian_matching(
        idx: int,
        processing_factory: RpcProcessFactory,
        matcher: HungarianMatcher,
        ego_gyro: np.array,
        params: dict[str, float], 
    ) -> tuple[list[RadarObject], FrameCluster, list[int]]:
    """
    Performs Hungarian Matching between the current frame and the previous frame, assiging an ID to each RadarObject found and tracking the ID across time.
    
    :param idx: Zero-based index of the frame to process.
    :type idx: int
    :param dbscan_config: The `dbscan` configuration object from the YAML configuration file.
    :type dbscan_config: dict[str, float]
    :param rpc_replay: Description
    :type rpc_replay: RpcReplay
    :param processing_factory: Description
    :type processing_factory: RpcProcessFactory
    :param matcher: Instantialized matcher object.
    :type matcher: HungarianMatcher
    :return: centroid, frame, labels
    :rtype: tuple[list[RadarObject], FrameCluster, list[int]]
    """
    
    # params = {
    #     "spatial_eps": dbscan_config.spatial_epsilon,
    #     "velocity_eps": dbscan_config.velocity_epsilon,
    #     "min_samples": dbscan_config.min_samples,
    #     "velocity_weight": dbscan_config.velocity_weight,
    #     "noise_velocity_threshold": dbscan_config.noise_velocity_threshold
    # }
    
    if idx < 1:
        """
        There is no previous frame to process. Return the processed clusters as it is.
        """
        centroid, frame, labels = processing_factory.get_processed_frame(idx=idx, **params)
        for i, obj in enumerate(centroid):
            obj.id = i
        return centroid, frame, labels

    moving_centroids_prev, _, _ = processing_factory.get_processed_frame(idx=idx - 1, **params)
    max_prev_id = 0
    for i, obj in enumerate(moving_centroids_prev):
        obj.id = i  # Mock ID
        max_prev_id = i
    
    moving_centroids_curr, processed_frame, valid_labels  = processing_factory.get_processed_frame(idx=idx, **params)
    
    
    ego_dx, ego_dy, ego_dtheta = _get_compensation_values(
        timestamp_curr=idx, 
        timestamp_prev=idx-1, 
        point_cloud=processed_frame.point_cloud, 
        ego_gyro=ego_gyro
    )
    detections_prev_compensated = copy.deepcopy(moving_centroids_prev)
    _compensate_motion(detections_prev_compensated, ego_dx, ego_dy, ego_dtheta)
    matches = matcher(detections_prev_compensated, moving_centroids_curr)
    
    assigned_current_indices = set()

    # Assign Previous Cluster's ID
    for curr_idx, prev_idx in matches.items():
        old_id = moving_centroids_prev[prev_idx].id
        moving_centroids_curr[curr_idx].id = old_id
        assigned_current_indices.add(curr_idx)
    next_new_id = max_prev_id + 1
    
    # Handle NEW Centroids
    for i, obj in enumerate(moving_centroids_curr):
        if i not in assigned_current_indices:
            obj.id = next_new_id
            next_new_id += 1

    return moving_centroids_curr, processed_frame, valid_labels
