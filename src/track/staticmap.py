from dataclasses import dataclass

import numpy as np


@dataclass
class EgoState:
    x: float      # Global X position of the car
    y: float      # Global Y position of the car
    yaw: float    # Heading in radians (0 = East, pi/2 = North)

def ego_to_global(ego_state: EgoState) -> np.ndarray:
    """
    Creates the Homogeneous Transformation Matrix (3x3)
    that converts points from the Vehicle Frame to the Global Frame.
    """
    c = np.cos(ego_state.yaw)
    s = np.sin(ego_state.yaw)
    
    # Rotation (R) + Translation (T)
    # [ cos -sin   x ]
    # [ sin  cos   y ]
    # [  0    0    1 ]
    transform_matrix = np.array([
        [c, -s, ego_state.x],
        [s,  c, ego_state.y],
        [0,  0, 1.0        ]
    ])
    return transform_matrix

def target_to_global(local_measurement: np.ndarray, ego_matrix: np.ndarray) -> np.ndarray:
    """
    Converts a single local measurement [x_rel, y_rel] to [x_global, y_global].
    """
    # 1. Convert local point to homogeneous coordinates: [x, y, 1]
    # We allow input to be 2D (pos) or 3D (pos+doppler)
    
    local_pos_homogeneous = np.array([local_measurement[0], local_measurement[1], 1.0])
    
    # 2. Apply Transformation
    global_pos_homogeneous = ego_matrix @ local_pos_homogeneous
    
    # 3. Extract just x, y
    global_x = global_pos_homogeneous[0]
    global_y = global_pos_homogeneous[1]
    
    # 4. Handle extra data (Doppler)
    # Note: Doppler is scalar and doesn't rotate, but it represents 
    # relative speed. For global tracking, we usually just pass it through
    # or leave it out if we only care about global position.
    if len(local_measurement) > 2:
        return np.array([global_x, global_y, local_measurement[2]])
    else:
        return np.array([global_x, global_y])