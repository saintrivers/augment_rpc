from pipepine.factory import RpcProcessFactory
from processing.association import HungarianMatcher
from processing.datareader import load_metadata, prepare_experiment_data
from processing.radarproc import RpcReplay

from joblib import Memory


location = './.cachedir'
memory = Memory(location, verbose=0)


@memory.cache
def hangarian_matching(config, rpc_replay: RpcReplay, processing_factory: RpcProcessFactory, matcher: HungarianMatcher):
    matchings = []
    for idx in range(1, rpc_replay.sim_length_steps):
        params = {
            "spatial_eps": config.dbscan.spatial_epsilon,
            "velocity_eps": config.dbscan.velocity_epsilon,
            "min_samples": config.dbscan.min_samples,
            "velocity_weight": config.dbscan.velocity_weight,
            "noise_velocity_threshold": config.dbscan.noise_velocity_threshold
        }
        moving_centroids_prev, _, _ = processing_factory.get_processed_frame(idx=idx - 1, **params)
        moving_centroids_curr, _, _ = processing_factory.get_processed_frame(idx=idx, **params)
        matched_output = matcher(moving_centroids_prev, moving_centroids_curr)
        matchings.append(matched_output)

        for curr_idx, prev_idx in matched_output.items():
            # 1. Get the ID of the old object
            old_id = moving_centroids_prev[prev_idx].id

            # 2. Assign it to the new object
            moving_centroids_curr[curr_idx].id = old_id
        
    return matchings


def main():
    """Main function to run the hyperparameter visualization."""
    
    ego_id, config = load_metadata(config_file="config/base.yml")
    rpc_replay = prepare_experiment_data(config.sim.datadir, ego_id)
    processing_factory = RpcProcessFactory(rpc_replay)

    matcher = HungarianMatcher(max_distance=2.0)
    matchings = hangarian_matching(config, rpc_replay, processing_factory, matcher)
    


if __name__ == "__main__":
    main()