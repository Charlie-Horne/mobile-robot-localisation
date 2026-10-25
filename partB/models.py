"""Particle filter sensor and motion model implementations.
"""

import numpy as np
from numpy import cos, sin, tan, arccos, arcsin, arctan2, sqrt, exp
from numpy.random import randn
from utils import gauss, wraptopi, angle_difference


def motion_model(particle_poses, speed_command, odom_pose, odom_pose_prev, dt):
    """Apply motion model and return updated array of particle_poses.

    Parameters
    ----------

    particle_poses: an M x 3 array of particle_poses where M is the
    number of particles.  Each pose is (x, y, theta) where x and y are
    in metres and theta is in radians.

    speed_command: a two element array of the current commanded speed
    vector, (v, omega), where v is the forward speed in m/s and omega
    is the angular speed in rad/s.

    odom_pose: the current local odometry pose (x, y, theta).

    odom_pose_prev: the previous local odometry pose (x, y, theta).

    dt is the time step (s).

    Returns
    -------
    An M x 3 array of updated particle_poses.

    """

    M = particle_poses.shape[0]
    
    # TODO.  For each particle calculate its predicted pose plus some
    # additive error to represent the process noise.  With this demo
    # code, the particles move in the -y direction with some Gaussian
    # additive noise in the x direction.  Hint, to start with do not
    # add much noise.

    dx_global = odom_pose[0] - odom_pose_prev[0]
    dy_global = odom_pose[1] - odom_pose_prev[1]
    dtheta = wraptopi(odom_pose[2] - odom_pose_prev[2])

    theta_prev = odom_pose_prev[2]
    local_dx = dx_global * cos(theta_prev) + dy_global * sin(theta_prev)
    local_dy = -dx_global * sin(theta_prev) + dy_global * cos(theta_prev)

    # Process noise standard deviations -- tune these by trial and error
    sigma_trans = 0.005   # noise on forward/sideways displacement (m)
    sigma_rot = 0.008      # noise on heading change (rad)

    for m in range(M):
        # Add noise to the local motion before applying it to this particle
        noisy_dx = local_dx + randn(1) * sigma_trans
        noisy_dy = local_dy + randn(1) * sigma_trans
        noisy_dtheta = dtheta + randn(1) * sigma_rot

        theta_p = particle_poses[m, 2]
        particle_poses[m, 0] += noisy_dx * cos(theta_p) - noisy_dy * sin(theta_p)
        particle_poses[m, 1] += noisy_dx * sin(theta_p) + noisy_dy * cos(theta_p)
        particle_poses[m, 2] = wraptopi(theta_p + noisy_dtheta)
    # print(f"sigma_trans={sigma_trans}, sigma_rot={sigma_rot}")
    return particle_poses


def sensor_model(particle_poses, beacon_pose, beacon_loc):
    """Apply sensor model and return particle weights.

    Parameters
    ----------
    
    particle_poses: an M x 3 array of particle_poses (in the map
    coordinate system) where M is the number of particles.  Each pose
    is (x, y, theta) where x and y are in metres and theta is in
    radians.

    beacon_pose: the measured pose of the beacon (x, y, theta) in the
    robot's camera coordinate system.

    beacon_loc: the pose of the currently visible beacon (x, y, theta)
    in the map coordinate system.

    Returns
    -------
    An M element array of particle weights.  The weights do not need to be
    normalised.

    """

    M = particle_poses.shape[0]
    particle_weights = np.zeros(M)
    
    # TODO.  For each particle calculate its weight based on its pose,
    # the relative beacon pose, and the beacon location.

    # Observed range and bearing from the actual sensor reading
    obs_range = sqrt(beacon_pose[0]**2 + beacon_pose[1]**2)
    obs_bearing = arctan2(beacon_pose[1], beacon_pose[0])

    sigma_range = 0.05    # metres, starting guess per the guide
    sigma_bearing = 0.1  # radians, starting guess per the guide

    for m in range(M):
        px, py, ptheta = particle_poses[m]

        # Predicted beacon position relative to this particle (inverse transform)
        dx = beacon_loc[0] - px
        dy = beacon_loc[1] - py
        pred_range = sqrt(dx**2 + dy**2)
        pred_bearing = wraptopi(arctan2(dy, dx) - ptheta)

        # Likelihood of the actual observation given this particle's prediction
        range_error = obs_range - pred_range
        bearing_error = angle_difference(obs_bearing, pred_bearing)

        weight = gauss(range_error, 0, sigma_range) * gauss(bearing_error, 0, sigma_bearing)
        particle_weights[m] = weight
    # print(f"sigma_range={sigma_range}, sigma_bearing={sigma_bearing}")
    return particle_weights