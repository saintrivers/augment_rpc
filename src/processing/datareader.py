import json
import os 
import numpy as np
from omegaconf import OmegaConf
import pandas as pd

from processing.radarproc import RpcReplay


def load_metadata(config_file: str = "config/base.yml"):
    config = OmegaConf.load(config_file)

    metadata_path = os.path.join(config.sim.datadir, "metadata.json") # type: ignore
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    ego_id = metadata["ego_vehicle_id"]
    return ego_id, config


def prepare_experiment_data(datadir: str, ego_id: int) -> RpcReplay:
    """
    Loads rpc data structure designed for easy fetching and indexing of recorded simulation data.
    
    :param str datadir: absolute or relative path to the dataset root
    :param int ego_id: id of the ego vehicle set by the carla simulator
    :return rpc_replay: RpcReplay object
    """
    rpc, frame_ids = load_radar_data(datadir)
    imu_df = load_ego_imu_data(f"{datadir}/imu_data.csv", ego_id)
    # nearby_df, ego_traj, frame_ids = load_and_prepare_data(config.sim.datadir, config.sim.ego_id)
    sensor_transforms = load_radar_config(datadir)
    rpc_replay = RpcReplay(rpc, frame_ids, sensor_transforms, imu_df)
    return rpc_replay


def load_radar_config(directory):
    """Loads radar transform and rotation configuration from a JSON file."""
    sensor_config = {}
    sensor_transforms = {}

    with open(os.path.join(directory, "radar_config.json"), 'r') as f:
        sensor_config = json.load(f)

    for name, _sensor_conf in sensor_config['sensors'].items():
        t = _sensor_conf['transform']
        sensor_transforms[name] = {
            'pos': np.array([t['x'], t['y'], t['z']]),
            'yaw_deg': t['yaw_deg']
        }
    return sensor_transforms


def load_radar_data(directory):
    """Loads and sorts radar data from front, left, and right subdirectories."""
    rpc = {}
    all_frame_ids = set()

    for sensor in ['front_center', 'front_left', 'front_right', 'back_left', 'back_right']:
        path = os.path.join(directory, sensor)
        if not os.path.isdir(path):
            print(f"Warning: Directory not found at {path}")
            rpc[sensor] = []
            continue
        
        # Sort filenames numerically to ensure chronological order
        filenames = sorted(
            [f for f in os.listdir(path) if f.endswith('.npy')],
            key=lambda f: int(os.path.splitext(f)[0])
        )
        rpc[sensor] = [np.load(os.path.join(path, filename)) for filename in filenames]
        # Collect frame IDs from the filenames
        all_frame_ids.update([int(os.path.splitext(f)[0]) for f in filenames])

    return rpc, sorted(list(all_frame_ids))


def load_ego_imu_data(filepath, ego_id):
    """
    Loads IMU data for the specified ego vehicle and sets the frame_id as the index.
    """
    imu_df = pd.read_csv(filepath)
    ego_imu_df = imu_df[imu_df['vehicle_id'] == ego_id].copy()
    ego_imu_df.set_index('frame_id', inplace=True)
    return ego_imu_df


def load_gt(datadir, ego_id):
    """Loads vehicle data, isolates the ego trajectory, and gets frame list."""
    # Construct file paths from arguments
    csv_nearby_path = os.path.join(datadir, "nearby_vehicles.csv")
    csv_all_path = os.path.join(datadir, "vehicle_coordinates.csv")

    nearby_df = pd.read_csv(csv_nearby_path)
    all_df = pd.read_csv(csv_all_path)

    # Get the Ego's trajectory for every frame so we can center the camera
    ego_traj = all_df[all_df["vehicle_id"] == ego_id].set_index("frame_id")[
        ["x", "y", "yaw"]
    ]

    # Get a list of all unique frames to loop through
    frames = sorted(nearby_df["frame_id"].unique())

    return nearby_df, ego_traj, frames
