from pipepine.core import ProcessingPipeline
from processing.clustering import DbscanClusterer, FrameCluster, PointCloudPreprocessor
from processing.radarproc import RpcReplay


class RpcProcessFactory:
    """
    A class that applies DBSCAN clustering to a RpcReplay object.

    This class acts as a factory for FrameCluster objects. It is initialized with
    the full radar data replay. When called, it constructs and runs a processing
    pipeline, caching the final result.
    """
    def __init__(self, rpc_replay: RpcReplay):
        """
        Initializes the RpcClusterFactory with data.

        ArgsSCAN `min_samples` parameter.
            velocity_weight (float): The weight for the velocity dimension.

        Returns:
            rpc_replay (RpcReplay): The pre-processed radar data sequence.
        """
        self.rpc_replay = rpc_replay
        self._memo = {}

    def get_processed_frame(self, idx: int, eps: float, min_samples: int, velocity_weight: float) -> FrameCluster:
        """
        Computes or retrieves the clustering result for a specific frame.

        Args:
            idx (int): The index of the frame to cluster.
            eps (float): The DBSCAN `eps` parameter.
            min_samples (int): The DBSCAN `min_samples` parameter.
            velocity_weight (float): The weight for the velocity dimension.

        Returns
            FrameCluster: An object containing the clustering results for the frame.
        """
        # Create a key that uniquely identifies this clustering request
        cache_key = (idx, eps, min_samples, velocity_weight)
        if cache_key in self._memo:
            return self._memo[cache_key]

        rpc_frame = self.rpc_replay[idx]

        # Define the processing pipeline
        pipeline = ProcessingPipeline(
            PointCloudPreprocessor(velocity_weight=velocity_weight),
            DbscanClusterer(eps=eps, min_samples=min_samples)
            # --- You can add more steps here! ---
            # e.g., BoundingBoxCalculator(), ClusterFilter(min_size=3)
        )

        # Run the pipeline and cache the result
        result = pipeline.run(rpc_frame)
        self._memo[cache_key] = result
        return result
