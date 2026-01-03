import os
import numpy as np
import argparse

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

from processing.visualization_data_provider import VisualizationDataProvider, MethodParams

def main():
    """Main function to run the web-based hyperparameter visualizer."""
    parser = argparse.ArgumentParser(
        description="Web-based visualizer for radar clustering with interactive hyperparameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--config', type=str, default="config/base.yml", help='Path to the YAML configuration file.')
    cli_args = parser.parse_args()

    # --- Centralized Data Loading ---
    data_provider = VisualizationDataProvider(cli_args.config)
    config = data_provider.config  # Get config for initial slider values
    print("Starting web server...")

    # --- Dash App Initialization ---
    app = dash.Dash(__name__)

    # --- App Layout ---
    app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif'}, children=[
        html.H1("Radar Point Cloud Analysis", style={'textAlign': 'center'}),
        html.Div(style={'display': 'flex', 'flexDirection': 'row'}, children=[
            # Left side: Graph
            html.Div(
                dcc.Graph(id='radar-plot', style={'height': '85vh'}),
                style={'flex': '3', 'padding': '10px'}
            ),
            # Right side: Control Panel
            html.Div(
                [
                    html.H4("Controls", style={'textAlign': 'center'}),
                    html.Div([
                        html.Label('Frame ID'),
                        dcc.Slider(
                            id='frame-slider',
                            min=0,
                            max=data_provider.rpc_replay.sim_length_steps,
                            value=0,
                            step=1,
                            marks={i: str(i) for i in range(0, data_provider.rpc_replay.sim_length_steps + 1, 50)}
                        ),
                    ], style={'padding': '15px 10px'}),
                    html.Div([
                        html.Label('Velocity Weight'),
                        dcc.Slider(
                            id='vel-weight-slider',
                            min=0, max=20, value=config.dbscan.velocity_weight, step=0.1
                        ),
                    ], style={'padding': '15px 10px'}),
                    html.Div([
                        html.Label('Spatial ε (m)'),
                        dcc.Slider(
                            id='spatial-eps-slider',
                            min=0.1, max=5, value=config.dbscan.spatial_epsilon, step=0.1
                        ),
                    ], style={'padding': '15px 10px'}),
                    html.Div([
                        html.Label('Velocity ε (m/s)'),
                        dcc.Slider(
                            id='velocity-eps-slider',
                            min=0.1, max=5, value=config.dbscan.velocity_epsilon, step=0.1
                        ),
                    ], style={'padding': '15px 10px'}),
                    html.Div([
                        html.Label('Min Samples'),
                        dcc.Slider(
                            id='min-samples-slider',
                            min=1, max=20, value=config.dbscan.min_samples, step=1
                        ),
                    ], style={'padding': '15px 10px'}),
                    html.Div([
                        html.Label('Noise Velocity Threshold (m/s)'),
                        dcc.Slider(
                            id='noise-vel-thresh-slider',
                            min=0.1, max=5, value=config.dbscan.noise_velocity_threshold, step=0.1
                        ),
                    ], style={'padding': '15px 10px'}),
                ],
                style={'flex': '1', 'padding': '20px', 'borderLeft': '1px solid #ccc', 'backgroundColor': '#f8f9fa'}
            )
        ])
    ])

    @app.callback(
        Output('radar-plot', 'figure'),
        [Input('frame-slider', 'value'),
         Input('vel-weight-slider', 'value'),
         Input('spatial-eps-slider', 'value'),
         Input('velocity-eps-slider', 'value'),
         Input('min-samples-slider', 'value'),
         Input('noise-vel-thresh-slider', 'value')]
    )
    def update_graph(frame_idx, vel_weight, spatial_eps, velocity_eps, min_samples, noise_vel_thresh):
        # --- Get Processed Data ---
        params = MethodParams(
            vel_weight=vel_weight,
            spatial_eps=spatial_eps,
            velocity_eps=velocity_eps,
            min_samples=min_samples,
            noise_velocity_threshold=noise_vel_thresh
        )
        moving_centroids, processed_frame, valid_labels, gt_frame = data_provider.get_frame_data(
            frame_idx, params
        )

        # --- Create Plotly Figure ---
        fig = go.Figure()

        # Ego vehicle
        fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers', marker=dict(symbol='triangle-up', color='red', size=15), name='Ego'))

        # Ground Truth
        if not gt_frame.other_vehicles.empty:
            fig.add_trace(go.Scatter(
                x=gt_frame.other_vehicles['y_relative'], 
                y=gt_frame.other_vehicles['x_relative'],
                mode='markers',
                marker=dict(symbol='square-open', color='yellow', size=12, line=dict(width=2)),
                name='Ground Truth'
            ))

        # Noise points
        valid_cluster_mask = np.isin(processed_frame.labels, valid_labels)
        all_noise_mask = processed_frame.noise_mask | (~valid_cluster_mask & processed_frame.cluster_mask)
        
        if np.any(all_noise_mask):
            fig.add_trace(go.Scatter(
                x=-processed_frame.point_cloud[all_noise_mask, 1], # Y-left
                y=processed_frame.point_cloud[all_noise_mask, 0], # X-fwd
                mode='markers',
                marker=dict(color='gray', size=4, opacity=0.5),
                name='Noise'
            ))

        # Clustered points
        if np.any(valid_cluster_mask):
            label_to_id_map = {label: obj.id for label, obj in zip(valid_labels, moving_centroids)}
            point_colors = [label_to_id_map.get(l, -1) for l in processed_frame.labels[valid_cluster_mask]]

            fig.add_trace(go.Scatter(
                x=-processed_frame.point_cloud[valid_cluster_mask, 1], # Y-left
                y=processed_frame.point_cloud[valid_cluster_mask, 0], # X-fwd
                mode='markers',
                marker=dict(
                    color=point_colors,
                    colorscale='Jet',
                    size=8,
                    showscale=False
                ),
                name='Clusters'
            ))

        # --- Layout and Styling ---
        view_radius = 80
        fig.update_layout(
            title=f"Frame ID: {frame_idx} | Valid Clusters: {len(valid_labels)}",
            xaxis_title="Left/Right Distance (m)",
            yaxis_title="Forward Distance (m)",
            xaxis=dict(range=[view_radius, -view_radius]),
            yaxis=dict(range=[-view_radius, view_radius]),
            # width and height are now controlled by the parent div's style
            margin=dict(l=40, r=40, t=80, b=40),
            showlegend=True,
            legend=dict(x=0.01, y=0.99),
            template='plotly_white'
        )
        # Enforce aspect ratio
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

        return fig

    # --- Run Server ---
    app.run(debug=True, host='0.0.0.0', port=8050)


if __name__ == "__main__":
    main()