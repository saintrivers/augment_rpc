import numpy as np
from scipy.optimize import linear_sum_assignment
from processing.clustering import RadarObject

class HungarianMatcher:
    def __init__(self, max_distance: float = 2.0):
        """
        Docstring for __init__
        
        :param self: Description
        :param max_distance: Sets the distance in meters (measured by the CARLA simulator) to be used for matching.
        :type max_distance: float
        """
        self.max_distance = max_distance

    def __call__(self, detections_prev: list[RadarObject], detections_curr: list[RadarObject]) -> dict[int, int]:
        num_prev = len(detections_prev)
        num_curr = len(detections_curr)

        if num_prev == 0 or num_curr == 0:
            return {}

        # 1. Create Cost Matrix (Vectorized for speed)
        # Extract centroids into arrays for fast broadcasting
        prev_cents = np.array([obj.centroid for obj in detections_prev])  # Shape: (N, 2)
        curr_cents = np.array([obj.centroid for obj in detections_curr])  # Shape: (M, 2)

        # Calculate Euclidean distance between all pairs
        # Result is (N, M) matrix where entry [i, j] is dist between prev[i] and curr[j]
        # Using linalg.norm on the difference
        diff = prev_cents[:, np.newaxis, :] - curr_cents[np.newaxis, :, :]
        cost_matrix = np.linalg.norm(diff, axis=2)

        # 2. Run Hungarian Algorithm (Munkres)
        # Note: We do NOT set np.inf here. We let it find the best geometric matches first.
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # 3. Filter Invalid Matches (Gating)
        associations = {}
        for r, c in zip(row_ind, col_ind):
            dist = cost_matrix[r, c]
            if dist < self.max_distance:
                # Only accept the match if it's within our physical threshold
                associations[c] = r 
            else:
                # If the "best" match is 50 meters away, it's not a match.
                # We ignore this, effectively treating it as a "Lost Track" for r 
                # and a "New Object" for c.
                pass

        return associations