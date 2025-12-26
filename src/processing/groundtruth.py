import pandas as pd
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
        ego_data = frame_data[frame_data['vehicle_id'] == self.ego_id]
        other_data = frame_data[frame_data['vehicle_id'] != self.ego_id]
        return FrameGroundTruth(ego_vehicle=ego_data, other_vehicles=other_data, all_vehicles=frame_data)