import os
import numpy as np
from dataclasses import dataclass, asdict

from processing.association import HungarianMatcher, hungarian_matching
from processing.datareader import load_metadata, prepare_experiment_data, load_ego_imu_data
from processing.groundtruth import GroundTruthReplay
from pipepine.factory import RpcProcessFactory
from processing.imu import get_gyro

@dataclass
class MethodParams:
    spatial_eps: float
    velocity_eps: float
    min_samples: int
    vel_weight: float
    noise_velocity_threshold: float


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

        self.ego_imu = load_ego_imu_data(f"{datadir}/imu_data.csv", self.ego_id)
        self.rpc_replay = prepare_experiment_data(datadir, self.ego_id)
        self.gt_replay = GroundTruthReplay(os.path.join(datadir, 'vehicle_coordinates.csv'), self.ego_id)
        self.processing_factory = RpcProcessFactory(self.rpc_replay)
        self.matcher = HungarianMatcher(max_distance=2.0)
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

        moving_centroids, processed_frame, valid_labels = hungarian_matching(
            idx=frame_idx, params=dbscan_config, processing_factory=self.processing_factory,
            matcher=self.matcher, ego_gyro=ego_gyro
        )

        # 2. Get Ground Truth
        gt_frame = self.gt_replay.get_frame_data(frame_idx)

        return moving_centroids, processed_frame, valid_labels, gt_frame