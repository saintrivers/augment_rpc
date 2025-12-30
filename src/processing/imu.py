import numpy as np

def get_gyro(ego_imu_df, frame_id: int):
    ego_gyro = ego_imu_df.loc[frame_id]
    return np.array(ego_gyro[['gyroscope_x', 'gyroscope_y', 'gyroscope_z']])
