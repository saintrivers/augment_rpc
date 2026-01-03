import os
import numpy as np
from dataclasses import dataclass, asdict

from processing.association import HungarianMatcher, get_compensation_values
from processing.clustering import RadarObject
from processing.datareader import load_metadata, prepare_experiment_data, load_ego_imu_data
from processing.groundtruth import GroundTruthReplay
from pipepine.factory import RpcProcessFactory
from processing.imu import get_gyro
from track.track import TrackManager

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

        self.ego_imu = load_ego_imu_data(f"{datadir}/imu_data.csv", self.ego_id)
        self.rpc_replay = prepare_experiment_data(datadir, self.ego_id)
        self.gt_replay = GroundTruthReplay(os.path.join(datadir, 'vehicle_coordinates.csv'), self.ego_id)
        self.processing_factory = RpcProcessFactory(self.rpc_replay)
        self.matcher = HungarianMatcher(max_distance=2.0)
        # Initialize the tracker with tunable parameters.
        # A higher process_noise makes the filter adapt more quickly to acceleration.
        self.track_manager = TrackManager(
            process_noise=1.0, gating_threshold=3.5
        )
        
        # State for ego-motion compensation
        self.ego_position = np.array([0.0, 0.0]) # x, y
        self.ego_yaw = 0.0 # radians
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
        # On the first frame, reset state
        if idx < 1:
            # Re-initialize to reset state, keeping the same parameters
            self.track_manager = TrackManager(
                process_noise=self.track_manager.process_noise,
                gating_threshold=self.track_manager.gating_threshold
            )
            self.ego_position = np.array([0.0, 0.0])
            self.ego_yaw = 0.0

        # 1. Get current raw detections (in current ego-vehicle frame)
        moving_centroids_curr, processed_frame, valid_labels  = self.processing_factory.get_processed_frame(idx=idx, **params)
        
        # 2. Calculate Ego Motion since last frame
        # We assume a constant time step (dt) of 0.1s (10 Hz)
        dt = 0.1
        dx, dy, dtheta = get_compensation_values(
            dt=dt,
            point_cloud=processed_frame.point_cloud,
            ego_gyro=ego_gyro
        )

        # 3. Update ego's global pose for the current frame
        # This must be done *before* compensating the measurements.
        # We need to rotate the incremental motion (dx, dy) by the *previous* yaw
        # to get the correct displacement in the global frame.
        c, s = np.cos(self.ego_yaw), np.sin(self.ego_yaw)
        self.ego_position[0] += c * dx - s * dy
        self.ego_position[1] += s * dx + c * dy
        self.ego_yaw += dtheta

        # 3. Compensate new measurements to move them into the global frame
        compensated_measurements = []
        for obj in moving_centroids_curr:
            # Start with the measurement in the current ego-frame
            p_ego = np.array([obj.centroid[0], obj.centroid[1]])

            # First, rotate the measurement by the ego's current total yaw
            c, s = np.cos(self.ego_yaw), np.sin(self.ego_yaw)
            R = np.array([[c, -s], [s, c]])
            p_world = R @ p_ego

            # Then, translate it by the ego's current total position
            p_world += self.ego_position

            compensated_measurements.append(np.array([p_world[0], p_world[1], obj.velocity]))

        # 4. Update the tracker with the compensated (global frame) measurements
        self.track_manager.update(compensated_measurements, dt=dt)

        # 4. Create the final list of RadarObjects from the tracker's state
        # This is the key step: we are now using the tracker as the source of truth.
        tracked_objects = []
        for track in self.track_manager.tracks:
            # The Kalman state has [px, py, vx, vy]. We take the position.
            pos_x, pos_y = track.state.x[0], track.state.x[1]
            
            # --- Transform Track back to Ego-Centric View for Visualization ---
            # Translate
            p_relative = np.array([pos_x, pos_y]) - self.ego_position
            # Rotate
            c, s = np.cos(-self.ego_yaw), np.sin(-self.ego_yaw)
            R_inv = np.array([[c, -s], [s, c]])
            p_final_ego_view = R_inv @ p_relative

            # Create a minimal RadarObject for visualization in the current ego-frame
            obj = RadarObject(
                centroid=(p_final_ego_view[0], p_final_ego_view[1], 0), 
                velocity=0, size=None, id=track.track_id
            )
            tracked_objects.append(obj)

        return tracked_objects, processed_frame, valid_labels