import numpy as np


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