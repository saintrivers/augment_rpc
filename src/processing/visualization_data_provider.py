import math
import os
import csv
import numpy as np
from dataclasses import dataclass, asdict

from processing.association import HungarianMatcher, get_compensation_values
from processing.clustering import BoxDimensions, RadarObject
from processing.datareader import load_metadata, prepare_experiment_data, load_ego_imu_data
from processing.groundtruth import GroundTruthReplay
from pipepine.factory import RpcProcessFactory
from processing.imu import get_gyro
from track.track import Track, TrackManager

@dataclass
class MethodParams:
    spatial_eps: float
    velocity_eps: float
    min_samples: int
    vel_weight: float
    noise_velocity_threshold: float

    def todict(self):
        return asdict(self)


class VisualizationDataProvider:
    """
    Handles data loading and processing for visualization scripts.
    This class centralizes the logic for fetching and processing radar data,
    acting as a single source of truth for different visualizers.
    """

    def __init__(self, config_path: str):
        """
        Initializes the data provider by loading all necessary datasets.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        print("Loading data for visualization...")
        self.ego_id, self.config = load_metadata(config_path)
        datadir = self.config.sim.datadir

        # Prepare CSV file for centroids output
        self.centroids_csv_path = os.path.join('centroids_output.csv')
        with open(self.centroids_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame_idx', 'track_id', 'x', 'y', 'z', 'velocity'])
        self.ego_imu = load_ego_imu_data(f"{datadir}/imu_data.csv", self.ego_id)
        self.rpc_replay = prepare_experiment_data(datadir, self.ego_id)
        self.gt_replay = GroundTruthReplay(os.path.join(datadir, 'vehicle_coordinates.csv'), self.ego_id)
        self.processing_factory = RpcProcessFactory(self.rpc_replay)
        self.matcher = HungarianMatcher(max_distance=2.0)
        # Initialize the tracker with tunable parameters.
        # A higher process_noise makes the filter adapt more quickly to acceleration.
        self.track_manager = TrackManager(
            process_noise=0.5, gating_threshold=5.0
        )
        
        # State for ego-motion compensation
        self.ego_position = np.array([0.0, 0.0]) # x, y
        self.ego_yaw = 0.0 # radians

        self.last_index = 1
        print("Data loaded successfully.")

    def get_frame_data(self, frame_idx: int, params: MethodParams):
        """
        Processes a single frame with the given parameters and returns all data needed for plotting.

        Args:
            frame_idx (int): The index of the frame to process.
            params (MethodParams): An object containing clustering hyperparameters.

        Returns:
            tuple: A tuple containing moving_centroids, processed_frame, valid_labels, and gt_frame.
        """
        # 1. Run Clustering and Association
        dbscan_config = {
            "spatial_eps": params.spatial_eps,
            "velocity_eps": params.velocity_eps,
            "min_samples": int(params.min_samples),
            "velocity_weight": params.vel_weight,
            "noise_velocity_threshold": params.noise_velocity_threshold,
        }

        target_frame_id = frame_idx + self.rpc_replay.start_frame_id
        ego_gyro = get_gyro(self.ego_imu, target_frame_id)

        # The TrackManager is now stateful, living in the class instance.
        # !todo: i find that the tracking ID gets dropped and a new one gets picked up very often
        # !todo: i also find that I need to highlight ghost positions if the track is not dropped yet
        moving_centroids, processed_frame, valid_labels = self._track(
            idx=frame_idx, 
            ego_gyro=ego_gyro,
            params=dbscan_config
        )

        # 2. Get Ground Truth
        gt_frame = self.gt_replay.get_frame_data(frame_idx)

        return moving_centroids, processed_frame, valid_labels, gt_frame
    
    def _track(self, idx: int, ego_gyro: np.ndarray, params: dict):
        """
        Processes a frame to get detections and updates the TrackManager.
        Returns a list of RadarObjects derived from the current state of the tracks.
        """
        
        # handle starting frame with no previous measurement
        if idx <= 1:
            relative_cluster_centroids, processed_frame, valid_labels  = self.processing_factory.get_processed_frame(idx=1, **params)
            for x in relative_cluster_centroids:
                x.id = self.last_index
                self.last_index += 1
            return relative_cluster_centroids, processed_frame, valid_labels

        relative_cluster_centroids, processed_frame, valid_labels  = self.processing_factory.get_processed_frame(idx=idx - 1, **params)
        
        measurements = np.array([[obj.centroid[0], obj.centroid[1], obj.velocity] for obj in relative_cluster_centroids])
        self.track_manager.update(measurements=measurements, dt=0.1)
        
        track_centroids = [track_to_centroid(track) for track in self.track_manager.tracks]
        print(f"Clusters: {len(relative_cluster_centroids)} - Tracks: {len(self.track_manager.tracks)}")

        # Write centroids to CSV file
        with open(self.centroids_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for obj in track_centroids:
                row = [idx, obj.id]
                row.extend(obj.centroid)
                row.append(obj.velocity)
                writer.writerow(row)

        return track_centroids, processed_frame, valid_labels

def track_to_centroid(track: Track) -> RadarObject:
    centroid = (track.state.x[0], track.state.x[1], 0)
    vx, vy = track.state.x[2], track.state.x[3]
    velocity = math.hypot(vx, vy)
    return RadarObject(centroid, velocity, BoxDimensions(1,1,1), track.track_id)