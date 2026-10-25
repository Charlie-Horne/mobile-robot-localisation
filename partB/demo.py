"""Particle filter demonstration program.

Note, this example is written in an attempt to be straightforward
to follow for those new to Python.  Normally, we would have written
it using about a dozen classes.
"""

from __future__ import print_function, division
try:
    import matplotlib
    matplotlib.use("TkAgg")
except ImportError:
    import matplotlib
    matplotlib.use("Qt5Agg")

from matplotlib.pyplot import figure, ion, ioff, show
from models import motion_model, sensor_model   
from utils import is_degenerate, resample
from plot import (plot_particles, plot_beacons, plot_path, keypress_handler,
                  plot_path_with_visibility, wait_until_key_pressed, get_key,
                  clean_poses)
from transform import find_transform, transform_pose
from numpy import genfromtxt, isnan, zeros, ones, sqrt
from numpy.random import uniform, seed


seed(1)

# Load data

# data is a (many x 13) matrix. Its columns are:
# time_ns, velocity_command, rotation_command, map_x, map_y, map_theta,
# odom_x, odom_y, odom_theta,
# beacon_ids, beacon_x, beacon_y, beacon_theta
data = genfromtxt('data.csv', delimiter=',', skip_header=1)

# Time in ns
t = data[:, 0]

# Velocity command in m/s, rotation command in rad/s
commands = data[:, 1:3]

# Position in map frame, from SLAM (this approximates ground truth)
slam_poses = data[:, 3:6]

# Position in odometry frame, from wheel encoders and gyro
odom_poses = data[:, 6:9]

# Id and measured position of beacon in camera frame
beacon_ids = data[:, 9]
beacon_poses = data[:, 10:13]
# Use beacon id of -1 if no beacon detected
beacon_ids[isnan(beacon_ids)] = -1
beacon_ids = beacon_ids.astype(int)
beacon_visible = beacon_ids >= 0

# map_data is a 16x13 matrix.  Its columns are:
# beacon_ids, x, y, theta, (9 columns of covariance)
map_data = genfromtxt('beacon_map.csv', delimiter=',', skip_header=1)

Nbeacons = map_data.shape[0]
beacon_locs = zeros((Nbeacons, 3))
for m in range(Nbeacons):
    id = int(map_data[m, 0])
    beacon_locs[id] = map_data[m, 1:4]

# Remove jumps in the pose history
slam_poses = clean_poses(slam_poses)

# Transform odometry poses into map frame
odom_to_map = find_transform(odom_poses[0], slam_poses[0])
odom_poses = transform_pose(odom_to_map, odom_poses)

ion()
fig = figure(figsize=(10, 5))
axes = fig.add_subplot(111)
fig.canvas.mpl_connect('key_press_event', keypress_handler)
fig.canvas.manager.full_screen_toggle()

plot_beacons(axes, beacon_locs, label='Beacons')
plot_path(axes, slam_poses, '-', label='SLAM')
# Uncomment to show odometry when debugging
# plot_path(axes, odom_poses, 'b:', label='Odom')

axes.legend(loc='lower right')

axes.set_xlim([-6, None])
axes.axis('equal')

# Tweak axes to make plotting better
axes.invert_yaxis()
axes.set_xlabel('y')
axes.set_ylabel('x')
axes.figure.canvas.draw()
axes.figure.canvas.flush_events()

# TODO: Set this to avoid twirl at start
# When your algorithm works well set to 0
start_step = 0

# TODO: Number of particles, you may need more or fewer!
Nparticles = 100

# TODO: How many steps between display updates
display_steps = 20

# TODO: Set initial belief.  This assumes a uniform distribution for the pose
# around the known starting pose.  It simplifies the localisation to a
# tracking problem.
start_pose = slam_poses[start_step]
# Spread particles across the whole mapped area instead of a tight window
# around the true starting pose, using beacon locations to bound the area
margin = 1.0  # extra metres beyond the outermost beacons
Xmin = beacon_locs[:, 0].min() - margin
Xmax = beacon_locs[:, 0].max() + margin
Ymin = beacon_locs[:, 1].min() - margin
Ymax = beacon_locs[:, 1].max() + margin
Tmin = -3.14159
Tmax = 3.14159

weights = ones(Nparticles)
poses = zeros((Nparticles, 3))

for m in range(Nparticles):
    poses[m] = (uniform(Xmin, Xmax),
                uniform(Ymin, Ymax),
                uniform(Tmin, Tmax))

Nposes = odom_poses.shape[0]
est_poses = zeros((Nposes, 3))

plot_particles(axes, poses, weights)
axes.set_title('Push space to start/stop, dot to move one step, q to quit...')
wait_until_key_pressed()

state = 'run'
display_step_prev = 0
for n in range(start_step + 1, Nposes):

    # TODO: write motion model function in models.py
    poses = motion_model(poses, commands[n-1], odom_poses[n],
                         odom_poses[n - 1], t[n] - t[n - 1])

    if beacon_visible[n]:

        beacon_id = beacon_ids[n]
        beacon_loc = beacon_locs[beacon_id]
        beacon_pose = beacon_poses[n]

        # TODO: write sensor model function in models.py
        weights *= sensor_model(poses, beacon_pose, beacon_loc)

        if sum(weights) < 1e-50:
            print('All weights are close to zero, you are lost...')
            # TODO: Do something to recover
            break

        if is_degenerate(weights):
            print('Resampling %d' % n)
            resample(poses, weights)

    est_poses[n] = poses.mean(axis=0)

    if (n > display_step_prev + display_steps) or state == 'step':
        print(n)

        # Show particle cloud
        plot_particles(axes, poses, weights)

        # Leave breadcrumbs showing current odometry
        # plot_path(axes, odom_poses[n], 'k.')

        # Show mean estimate
        plot_path_with_visibility(axes,
                                  est_poses[display_step_prev - 1: n + 1],
                                  '-',
                                  visibility=beacon_visible[display_step_prev - 1: n + 1])
        display_step_prev = n

        print(state)

    key = get_key()
    if key == '.':
        state = 'step'
    elif key == ' ':
        if state == 'run':
            state = 'pause'
        else:
            state = 'run'

    if state == 'pause':
        wait_until_key_pressed()
    elif state == 'step':
        wait_until_key_pressed()

# Compute error against SLAM ground truth
errors = []
for n in range(start_step, Nposes):
    dx = est_poses[n, 0] - slam_poses[n, 0]
    dy = est_poses[n, 1] - slam_poses[n, 1]
    dist = sqrt(dx**2 + dy**2)
    errors.append(dist)

# Ignore any NaN values (SLAM sometimes has bad/missing data points)
valid_errors = [e for e in errors if not (e != e)]  # e != e is True only for NaN

mean_error = sum(valid_errors) / len(valid_errors)
max_error = max(valid_errors)
final_error = errors[-1]

print(f"Mean error: {mean_error:.4f} m")
print(f"Max error:  {max_error:.4f} m")
print(f"Final error: {final_error:.4f} m")

# Display final plot
print('Done, displaying final plot')
ioff()
show()

# Save final plot to file
plot_filename = 'path.pdf'
print('Saving final plot to', plot_filename)

plot_path(axes, est_poses, 'r-', label='PF')
axes.legend(loc='lower right')

fig = figure(figsize=(10, 5))
axes = fig.add_subplot(111)

plot_beacons(axes, beacon_locs, label='Beacons')
plot_path(axes, slam_poses, 'b-', label='SLAM')
plot_path(axes, odom_poses, 'b:', label='Odom')
plot_path(axes, est_poses, 'r-', label='PF')
axes.legend(loc='lower right')

axes.set_xlim([-6, None])
axes.axis('equal')

# Tweak axes to make plotting better
axes.invert_yaxis()
axes.set_xlabel('y')
axes.set_ylabel('x')
fig.savefig(plot_filename, bbox_inches='tight')
