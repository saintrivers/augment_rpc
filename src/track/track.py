import numpy as np

from track.kalman_filter import kf_predict, kf_update
from track.kalman_filter import KalmanState
from scipy.optimize import linear_sum_assignment

class Track:
    def __init__(self, track_id, initial_measurement):
        self.track_id = track_id
        # Initialize state with measurement, velocity = 0
        self.state = KalmanState(
            x=np.array([initial_measurement[0], initial_measurement[1], 0, 0]),
            P=np.eye(4)
        )
        self.skipped_frames = 0 # For counting "invisibility"

class TrackManager:
    def __init__(self, process_noise: float = 0.1, gating_threshold: float = 2.0):
        self.tracks: list[Track] = []
        self.next_id = 0
        self.max_skipped_frames = 20  # From the paper
        self.process_noise = process_noise
        self.gating_threshold = gating_threshold   # Max distance to consider a match valid

    def update(self, measurements, dt):
        """
        Main loop:
        1. Predict all tracks.
        2. Match predictions to new DBSCAN measurements (Hungarian Algo).
        3. Update matched tracks.
        4. Create new tracks / Delete dead tracks.
        
        :param measurements: A Python list of np.arrays with shape (N, 2), (N, 3) or (N, 4). Example: [(10.0,4.0),(10.0,4.0),(10.0,4.0)]
        :type measurements: np.array
        :param dt: Description
        :type dt: Description
        """
        
        # --- 1. PREDICT ---
        # Every track predicts its new position regardless of measurements
        for track in self.tracks:
            track.state = kf_predict(track.state, dt, self.process_noise)

        # --- 2. ASSOCIATION (The Hungarian Algorithm) ---
        
        # If no measurements come in, all tracks are unmatched
        if len(measurements) == 0:
            for track in self.tracks:
                track.skipped_frames += 1
            self._cleanup_tracks()
            return

        # Create Cost Matrix (Euclidean Distance)
        # Rows = Existing Tracks, Cols = New Measurements
        cost_matrix = np.zeros((len(self.tracks), len(measurements)))
        
        for t_idx, track in enumerate(self.tracks):
            for m_idx, meas in enumerate(measurements):
                # Always use the first two elements (position) for distance calculation
                # This works for measurements of size 2, 3, or 4.
                predicted_pos = track.state.x[:2]
                measurement_pos = meas[:2]
                
                # Calculate Euclidean distance based on position only
                dist = np.linalg.norm(predicted_pos - measurement_pos)
                
                cost_matrix[t_idx, m_idx] = dist

        # Apply Hungarian Algorithm
        # row_indices -> indices of tracks
        # col_indices -> indices of measurements
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # Sets to keep track of what we processed
        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_measurements = set(range(len(measurements)))

        # --- 3. HANDLING MATCHES ---
        for r, c in zip(row_indices, col_indices):
            
            # GATING: Just because Hungarian matched them doesn't mean it's right.
            # If the distance is too huge, reject the match.
            if cost_matrix[r, c] < self.gating_threshold:
                
                # VALID MATCH: Update the specific track with the specific measurement
                track = self.tracks[r]
                measurement = measurements[c]
                
                # Call the Update step
                track.state = kf_update(track.state, measurement)
                
                # Reset invisibility counter
                track.skipped_frames = 0
                
                # Remove from "unmatched" sets
                unmatched_tracks.discard(r)
                unmatched_measurements.discard(c)
            else:
                # Matched but too far away (treat as unmatched)
                pass 

        # --- 4. HANDLING UNMATCHED ITEMS ---
        
        # A. Unmatched Tracks (Sensor didn't see them)
        for t_idx in unmatched_tracks:
            self.tracks[t_idx].skipped_frames += 1
            # Note: We do NOT call kf_update here. 
            # The track keeps the "Predicted" state from step 1.
        
        # B. Unmatched Measurements (New objects entering the frame)
        for m_idx in unmatched_measurements:
            self._create_new_track(measurements[m_idx])

        # C. Remove dead tracks
        self._cleanup_tracks()

    def _create_new_track(self, measurement):
        new_track = Track(self.next_id, measurement)
        self.tracks.append(new_track)
        self.next_id += 1
        print(f"Created new track ID: {new_track.track_id}")

    def _cleanup_tracks(self):
        # Remove tracks that have been invisible for too long
        self.tracks = [t for t in self.tracks if t.skipped_frames < self.max_skipped_frames]