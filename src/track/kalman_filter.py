from dataclasses import dataclass
import numpy as np


@dataclass
class KalmanState:
    """
    A data class to hold the state and covariance for a single tracked object.

    Attributes:
        x (np.ndarray): The state vector [px, py, vx, vy].
        P (np.ndarray): The 4x4 covariance matrix of the state estimate.
    """
    x: np.ndarray
    P: np.ndarray


def kf_predict(state: KalmanState, dt: float, process_noise: float) -> KalmanState:
    """Step 1: Predict where the object will be based on motion."""
    F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    # Process Noise (Q)
    G = np.array([[0.5 * dt**2], [0.5 * dt**2], [dt], [dt]])
    Q = G @ G.T * process_noise**2

    # Predict state and covariance
    x_pred = F @ state.x
    P_pred = F @ state.P @ F.T + Q
    
    return KalmanState(x=x_pred, P=P_pred)


def kf_update(state: KalmanState, measurement: np.ndarray, measurement_noise: float = 0.5) -> KalmanState:
    """
    Unified EKF Update step that handles variable measurement sizes:
    - Size 2: [px, py]
    - Size 3: [px, py, doppler_velocity] (Requires Jacobian)
    - Size 4: [px, py, vx, vy]
    """
    
    meas_dim = len(measurement)
    
    # Initialize variables that differ by case
    H = None
    R = None
    y_residual = None
    
    # --- CASE 1: Position Only [px, py] ---
    if meas_dim == 2:
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        R = np.eye(2) * measurement_noise**2
        
        # Linear residual: z - Hx
        y_residual = measurement - (H @ state.x)
        
    # --- CASE 2: Position + Doppler [px, py, v_radial] ---
    elif meas_dim == 3:
        px, py = state.x[0], state.x[1]
        vx, vy = state.x[2], state.x[3]
        
        # Calculate Range (r) for projection
        r = np.sqrt(px**2 + py**2)
        if r < 1e-3: r = 1e-3 # Avoid divide by zero
        
        # Jacobian Matrix (Linearized approximation)
        # Row 3 projects vx, vy onto the radial vector
        H = np.array([
            [1, 0, 0,   0],
            [0, 1, 0,   0],
            [0, 0, px/r, py/r] 
        ])
        
        # Measurement Noise
        # Typically Doppler is noisier or different than position, 
        # but here we scale it based on the input noise.
        R = np.diag([measurement_noise**2, measurement_noise**2, (measurement_noise * 2)**2])
        
        # Non-Linear Residual: z - h(x)
        # We must calculate the expected doppler explicitly using the non-linear equation
        expected_doppler = (vx * px/r) + (vy * py/r)
        expected_measurement = np.array([px, py, expected_doppler])
        
        y_residual = measurement - expected_measurement

    # --- CASE 3: Full State [px, py, vx, vy] ---
    elif meas_dim == 4:
        H = np.eye(4)
        R = np.eye(4) * measurement_noise**2
        
        # Linear residual: z - Hx
        y_residual = measurement - (H @ state.x)
        
    else:
        raise ValueError("Measurement must be size 2, 3, or 4")

    # --- Standard Kalman Update Steps (Identical for all cases) ---
    
    # 1. Calculate Kalman Gain
    S = H @ state.P @ H.T + R
    K = state.P @ H.T @ np.linalg.inv(S)

    # 2. Update State Estimate
    x_new = state.x + (K @ y_residual)
    
    # 3. Update Covariance Matrix
    P_new = (np.eye(4) - K @ H) @ state.P
    
    return KalmanState(x=x_new, P=P_new)