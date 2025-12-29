import numpy as np
import pandas as pd

def spherical_to_cartesian(points):
    """
    Converts radar points from spherical to Cartesian coordinates.
    The input format is [velocity, azimuth, altitude, depth].
    Returns (x, y, z, velocity).
    """
    velocity = points[:, 0]
    azimuth = points[:, 1]
    altitude = points[:, 2]
    depth = points[:, 3]

    x = depth * np.cos(altitude) * np.cos(azimuth)
    y = depth * np.cos(altitude) * np.sin(azimuth)
    z = depth * np.sin(altitude)
    return x, y, z, velocity


def transform_sensor_points(points_batch, sensor_transform):
    """
    Transforms a batch of sensor points from the sensor's local frame to the ego vehicle's frame.

    Args:
        points_batch (tuple): A tuple containing (x_local, y_local, z_local, velocity) numpy arrays.
        sensor_transform (dict): A dictionary with the sensor's 'pos' (np.array) and 'yaw_deg' (float).

    Returns:
        tuple: A tuple containing (x_vehicle, y_vehicle, z_vehicle) numpy arrays in the vehicle's coordinate system.
    """
    x_local, y_local, z_local, _ = points_batch
    
    # --- 1. Apply Rotation based on sensor's yaw ---
    yaw_rad = np.deg2rad(sensor_transform['yaw_deg'])
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    x_rotated = x_local * c - y_local * s
    y_rotated = x_local * s + y_local * c

    # --- 2. Apply Translation based on sensor's position ---
    position = sensor_transform['pos']
    x_vehicle = x_rotated + position[0]
    y_vehicle = y_rotated + position[1]
    z_vehicle = z_local + position[2]  # Z is unaffected by yaw

    return x_vehicle, y_vehicle, z_vehicle


def estimate_ego_velocity(points_batch, ego_gyro):
    """
    Estimates the ego vehicle's translational velocity using radar point data.

    This function solves a robust least-squares problem to find the ego
    velocity that best explains the observed Doppler velocities, after compensating
    for the velocity component induced by the ego vehicle's own rotation.

    The underlying equation is: v_doppler = v_rot_los - v_ego_los

    Args:
        points_batch (tuple): A tuple of (x, y, z, raw_velocities) numpy arrays.
                                Positions are in the vehicle frame.
        ego_gyro (np.array): The ego vehicle's gyroscope data [x, y, z] in rad/s.

    Returns:
        np.array: The estimated ego velocity vector [vx, vy, vz] in m/s.
    """
    x, y, z, raw_velocities = points_batch

    r = np.stack((x, y, z), axis=-1)
    ranges = np.linalg.norm(r, axis=1)
    ranges[ranges < 1e-6] = 1e-6
    r_hat = r / ranges[:, None]

    omega = -ego_gyro
    v_rot = np.cross(omega, r)
    v_rot_los = np.einsum('ij,ij->i', v_rot, r_hat)

    b = -(raw_velocities - v_rot_los)
    A = r_hat  # N x 3

    # Robust least squares
    v_ego, *_ = np.linalg.lstsq(A, b, rcond=None)
    return v_ego



def correct_velocities_for_translation(points_batch, ego_velocity):
    """
    Corrects radar Doppler velocities for ego vehicle translation.

    Args:
        points_batch (tuple):
            (x, y, z, raw_velocities), positions in meters (vehicle frame)
        ego_velocity (np.array):
            [vx, vy, vz] in m/s (vehicle frame)

    Returns:
        np.array:
            Doppler velocities corrected for ego translation
    """
    x, y, z, raw_velocities = points_batch

    # Point vectors
    point_vectors = np.stack((x, y, z), axis=-1)

    # Line-of-sight unit vectors
    ranges = np.linalg.norm(point_vectors, axis=1)
    ranges[ranges < 1e-6] = 1e-6
    unit_vectors = point_vectors / ranges[:, None]

    # Doppler induced by ego translation
    # v = -v_ego · r_hat
    v_translation = np.einsum('ij,j->i', unit_vectors, ego_velocity)

    # Remove it
    return raw_velocities + v_translation


def correct_velocities_for_ego_motion(points_batch, ego_gyro):
    """
    Corrects radial velocities for the ego vehicle's rotational motion.

    Args:
        points_batch (tuple): A tuple of (x, y, z, velocity) arrays in the vehicle's frame.
        ego_gyro (np.array): The ego vehicle's gyroscope data [x, y, z] in rad/s.

    Returns:
        np.array: The corrected radial velocities.
    """
    x, y, z, raw_velocities = points_batch
    
    # Point vectors from the vehicle's origin
    point_vectors = np.stack((x, y, z), axis=-1)
    
    # The angular velocity vector (omega) of the ego vehicle.
    omega = -ego_gyro # rad/s

    # Velocity induced by rotation: v_rot = omega x r
    velocity_from_rotation = np.cross(omega, point_vectors)

    # Project this velocity onto the line-of-sight vector for each point
    # The line-of-sight unit vector is the normalized point_vector
    norm = np.linalg.norm(point_vectors, axis=1)
    # Avoid division by zero for points at the origin
    # norm[norm == 0] = 1 
    norm[norm < 1e-6] = 1e-6
    unit_vectors = point_vectors / norm[:, np.newaxis]
    
    # Dot product of v_rot and unit_vector gives the component of rotational velocity along the line of sight
    # We use einsum for a vectorized dot product: sum(A[i,j] * B[i,j]) over j for each i
    v_correction = np.einsum('ij,ij->i', velocity_from_rotation, unit_vectors)

    return raw_velocities - v_correction

def collect_transformed_rpc(rpc, frame_idx: int, sensor_transforms: dict, ego_imu_record: dict = None):
    """
    Collects, transforms, and corrects radar points from all sensors for a single frame.

    This function iterates through all available sensors for a given frame index,
    performs the following steps for each sensor's data:
    1. Converts radar points from spherical to Cartesian coordinates.
    2. Transforms points from the sensor's local frame to the ego vehicle's frame.
    3. If IMU data is provided, corrects the Doppler velocities for the ego's
        rotational and translational motion.
    4. Aggregates the processed points from all sensors into a single point cloud.

    Args:
        rpc (dict): The raw radar point cloud data, keyed by sensor name.
        frame_idx (int): The index of the frame to process.
        sensor_transforms (dict): A dictionary containing the position and yaw for each sensor.
        ego_imu_record (dict, optional): A dictionary-like object with the ego vehicle's
                                            IMU data for this frame. Defaults to None.

    Returns:
        tuple: A tuple of lists (xs, ys, zs, velocities) for the aggregated point cloud.
    """
    xs, ys, zs, velocities = [], [], [], []
        
    for sensor_name, transform_info in sensor_transforms.items():
        if sensor_name in rpc and frame_idx < len(rpc[sensor_name]):
            # Get raw points in sensor-local coordinates
            points_cartesian_local = spherical_to_cartesian(rpc[sensor_name][frame_idx])
            raw_velocities = points_cartesian_local[3]
            
            # Transform points to the vehicle's coordinate frame
            x_vehicle, y_vehicle, z_vehicle = transform_sensor_points(points_cartesian_local, transform_info)
            
            # If ego vehicle's gyroscope data is provided, correct the velocities
            if ego_imu_record is not None:
                points_in_vehicle_frame = (x_vehicle, y_vehicle, z_vehicle, raw_velocities)
                ego_gyro = np.array([ego_imu_record['gyroscope_x'], ego_imu_record['gyroscope_y'], ego_imu_record['gyroscope_z']])
                corrected_velocities = correct_velocities_for_ego_motion(points_in_vehicle_frame, ego_gyro)
                
                points_in_vehicle_frame = (x_vehicle, y_vehicle, z_vehicle, corrected_velocities)
                x_v = estimate_ego_velocity(points_in_vehicle_frame, ego_gyro)
                corrected_velocities = correct_velocities_for_translation(points_in_vehicle_frame, x_v)
            else:
                # Otherwise, use the raw velocities
                corrected_velocities = raw_velocities
            
            xs.extend(x_vehicle); ys.extend(y_vehicle); zs.extend(z_vehicle)
            velocities.extend(corrected_velocities)
    return xs, ys, zs, velocities


def process_rpc_frame(rpc_frame: dict, sensor_transforms: dict, ego_imu_record: dict = None):
    """
    Processes a single frame of radar data from multiple sensors.

    Args:
        rpc_frame (dict): A dictionary where keys are sensor names and values are the radar data for a single frame.
        sensor_transforms (dict): A dictionary with sensor transform information.
        ego_imu_record (dict, optional): Ego vehicle IMU data for motion correction. Defaults to None.

    Returns:
        tuple: A tuple of (xs, ys, zs, velocities) for all aggregated points in the vehicle frame.
    """
    xs, ys, zs, velocities = [], [], [], []

    for sensor_name, transform_info in sensor_transforms.items():
        if sensor_name in rpc_frame:
            points_cartesian_local = spherical_to_cartesian(rpc_frame[sensor_name])
            raw_velocities = points_cartesian_local[3]

            x_vehicle, y_vehicle, z_vehicle = transform_sensor_points(points_cartesian_local, transform_info)

            if ego_imu_record is not None:
                points_in_vehicle_frame = (x_vehicle, y_vehicle, z_vehicle, raw_velocities)
                ego_gyro = np.array([ego_imu_record['gyroscope_x'], ego_imu_record['gyroscope_y'], ego_imu_record['gyroscope_z']])
                corrected_velocities = correct_velocities_for_ego_motion(points_in_vehicle_frame, ego_gyro)

                points_in_vehicle_frame = (x_vehicle, y_vehicle, z_vehicle, corrected_velocities)
                v_ego = estimate_ego_velocity(points_in_vehicle_frame, ego_gyro)
                corrected_velocities = correct_velocities_for_translation(points_in_vehicle_frame, v_ego)
            else:
                corrected_velocities = raw_velocities

            xs.extend(x_vehicle); ys.extend(y_vehicle); zs.extend(z_vehicle)
            velocities.extend(corrected_velocities)

    return xs, ys, zs, velocities

class RpcFrame:
    """
    A data class to hold the processed radar point cloud for a single frame.

    Attributes:
        x (np.ndarray): Array of X-coordinates in the vehicle frame (forward).
        y (np.ndarray): Array of Y-coordinates in the vehicle frame (right).
        z (np.ndarray): Array of Z-coordinates in the vehicle frame (up).
        velocities (np.ndarray): Array of corrected Doppler velocities for each point.
    """
    def __init__(self, x: list[float], y: list[float], z: list[float], velocity: list[float]):
        self.x = np.array(x)
        self.y = np.array(y)
        self.z = np.array(z)
        self.velocities = np.array(velocity)


class RpcReplay:
    """
    Pre-processes and stores an entire sequence of radar data for easy replay.

    This class loads all radar frames from a simulation, processes them into the
    ego vehicle's coordinate frame, corrects for ego motion, and stores them
    in memory indexed by a continuous frame sequence.

    Attributes:
        xs (list[list[float]]): List of X-coordinates for each frame.
        ys (list[list[float]]): List of Y-coordinates for each frame.
        zs (list[list[float]]): List of Z-coordinates for each frame.
        velocities (list[list[float]]): List of corrected velocities for each frame.
    """
    def _reindex(self, frame_ids: list[int]) -> list[int]:
        """
        Creates a continuous range of frame indices from the start to the end of a simulation.
        Assumes the input frame_ids list is sorted.
        """
        self.sim_length_steps = frame_ids[-1] - frame_ids[0]
        self.start_frame_id = frame_ids[0]
        return range(0, self.sim_length_steps+1)

    def __init__(self, rpc: dict, frame_ids: list[int], sensor_transforms: dict, imu_df: 'pd.DataFrame'):
        """
        Initializes the RpcReplay object by processing all radar frames.

        Args:
            rpc (dict): Raw radar data, keyed by sensor name.
            frame_ids (list[int]): A sorted list of all frame IDs in the simulation.
            sensor_transforms (dict): Configuration for sensor positions and orientations.
            imu_df (pd.DataFrame): A DataFrame containing IMU data for the ego vehicle, indexed by frame_id.
        """
        self.xs, self.ys, self.zs, self.velocities = [], [], [], []
        indices = self._reindex(frame_ids)

        for idx in indices: 
            try:
                imu_record = imu_df.loc[idx].to_dict()
            except KeyError:
                imu_record = None # No IMU data for this frame
            x, y, z, v = collect_transformed_rpc(rpc, idx, sensor_transforms, imu_record)
            x = np.array(x); y = np.array(y); z = np.array(z); v = np.array(v)
            self.xs.append(x); self.ys.append(y); self.zs.append(z); self.velocities.append(v)
    
    def __getitem__(self, idx: int) -> RpcFrame:
        """
        Makes the class indexable, retrieving the processed point cloud for a
        specific frame by its ordinal index (e.g., `replay[0]`).

        Args:
            idx (int): The zero-based index of the frame to retrieve.

        Returns:
            A tuple containing (xs, ys, zs, velocities) for the requested frame.
        """
        # return self.xs[idx], self.ys[idx], self.zs[idx], self.velocities[idx]
        return RpcFrame(self.xs[idx], self.ys[idx], self.zs[idx], self.velocities[idx])
    
    def get_frame_by_id(self, frame_id: int) -> RpcFrame:
        """
        Docstring for get_frame_by_id
        
        :param self: Description
        :param frame_id: Description
        :type frame_id: int
        :return: Description
        :rtype: RpcFrame
        """
        idx = frame_id - self.start_frame_id
        return self[idx]