import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class FrameGroundTruth:
    """
    A container for ground truth data for a single frame.
    
    Attributes:
        ego_vehicle (pd.DataFrame): DataFrame containing the row for the ego vehicle.
        other_vehicles (pd.DataFrame): DataFrame containing rows for all other vehicles.
        all_vehicles (pd.DataFrame): The complete data for the frame.
    """
    ego_vehicle: pd.DataFrame
    other_vehicles: pd.DataFrame
    all_vehicles: pd.DataFrame


class GroundTruthReplay:
    """
    Pre-processes and stores ground truth vehicle coordinate data for easy replay.
    """
    def __init__(self, csv_path: str, ego_id: int):
        """
        Initializes the replay object by loading vehicle coordinates from a CSV.

        Args:
            csv_path (str): The path to the vehicle_coordinates.csv file.
            ego_id (int): The ID of the ego vehicle.
        """
        self.ego_id = ego_id
        try:
            self.df = pd.read_csv(csv_path, index_col='frame_id')
        except FileNotFoundError:
            print(f"Error: Ground truth file not found at {csv_path}")
            self.df = pd.DataFrame()
        
        self.frames = sorted(self.df.index.unique()) if not self.df.empty else []

    def get_frame_data(self, frame_id: int) -> FrameGroundTruth:
        frame_data = self.df.loc[self.df.index == frame_id]
        ego_df = frame_data[frame_data['vehicle_id'] == self.ego_id]
        other_df = frame_data[frame_data['vehicle_id'] != self.ego_id].copy() # Use copy to avoid SettingWithCopyWarning

        if not ego_df.empty and not other_df.empty:
            ego_state = ego_df.iloc[0]
            ego_x, ego_y, ego_yaw_deg = ego_state['x'], ego_state['y'], ego_state['yaw']

            # --- 1. Translation: Move origin to ego vehicle ---
            relative_x = other_df["x"].values - ego_x
            relative_y = other_df["y"].values - ego_y

            # --- 2. Rotation: Align axes with ego vehicle's yaw ---
            # We apply a negative rotation to bring world coordinates into the ego's frame.
            yaw_rad = -np.deg2rad(ego_yaw_deg)
            c, s = np.cos(yaw_rad), np.sin(yaw_rad)

            # 'x_relative' is forward distance, 'y_relative' is lateral distance (left is positive)
            other_df['x_relative'] = relative_x * c - relative_y * s
            other_df['y_relative'] = -(relative_x * s + relative_y * c)

        else:
            # If there's no ego or no other vehicles, the relative columns will be empty
            other_df['x_relative'] = []
            other_df['y_relative'] = []

        return FrameGroundTruth(ego_vehicle=ego_df, other_vehicles=other_df, all_vehicles=frame_data)