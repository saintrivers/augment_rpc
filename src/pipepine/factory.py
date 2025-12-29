from pipepine.core import ProcessingPipeline
from processing.clustering import ClusterAnalyzer, MdDbscanClusterer, FrameCluster, PointCloudPreprocessor, RadarObject
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

    def get_processed_frame(
        self, 
        idx: int, 
        spatial_eps: 
        float, 
        velocity_eps: float, 
        min_samples: int, 
        velocity_weight: float,
        noise_velocity_threshold: float,
    ) -> tuple[list[RadarObject], FrameCluster, list[int]]:
        """
        Computes or retrieves the clustering result for a specific frame.

        Args:
            idx (int): The index of the frame to cluster.
            spatial_eps (float): The spatial distance tolerance for clustering (meters).
            velocity_eps (float): The velocity tolerance for clustering (m/s).
            min_samples (int): The DBSCAN `min_samples` parameter.
            velocity_weight (float): The weight for the velocity dimension.

        Returns
            A tuple containing (list of RadarObjects, FrameCluster result, list of valid labels).
        """
        # Create a key that uniquely identifies this clustering request
        cache_key = (idx, spatial_eps, velocity_eps, min_samples, velocity_weight)
        if cache_key in self._memo:
            return self._memo[cache_key]

        rpc_frame = self.rpc_replay[idx]

        # Define the processing pipeline
        pipeline = ProcessingPipeline(
            PointCloudPreprocessor(velocity_weight=velocity_weight), # This step is still needed by MdDbscanClusterer
            MdDbscanClusterer(spatial_eps=spatial_eps, velocity_eps=velocity_eps, min_samples=min_samples),
            ClusterAnalyzer(rpc_frame, noise_velocity_threshold),
            # --- You can add more steps here! ---
            # e.g., BoundingBoxCalculator(), ClusterFilter(min_size=3)
        )

        # Run the pipeline and cache the result
        result = pipeline.run(rpc_frame)
        self._memo[cache_key] = result
        return result
