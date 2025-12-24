import numpy as np


def transform_to_ego_view(neighbor_df, ego_x, ego_y, ego_yaw_deg) -> np.ndarray:
    """
    Transforms world coordinates of neighbor vehicles to the ego vehicle's relative coordinate system.

    Args:
        neighbor_df (pd.DataFrame): DataFrame of neighbor vehicles for a single frame.
                                    Must contain 'x' and 'y' columns.
        ego_x (float): Ego vehicle's world x-coordinate.
        ego_y (float): Ego vehicle's world y-coordinate.
        ego_yaw_deg (float): Ego vehicle's world yaw in degrees.

    Returns:
        np.ndarray: An N x 2 array of transformed [left, forward] coordinates for plotting.
    """
    # 1. Get relative positions of neighbors to the ego vehicle
    relative_x = neighbor_df["x"].values - ego_x
    relative_y = neighbor_df["y"].values - ego_y

    # 2. Rotate these relative positions based on the ego's yaw
    # We apply a negative rotation to align the world with the ego's perspective.
    yaw_rad = -np.deg2rad(ego_yaw_deg)
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)

    # rotated_x is forward, rotated_y is lateral (left is positive)
    rotated_x = relative_x * c - relative_y * s
    rotated_y = -(relative_x * s + relative_y * c)

    return np.column_stack([rotated_y, rotated_x]) # Plot left on x-axis, forward on y-axis
