from numpy import loadtxt, std, abs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots, show
from scipy.optimize import curve_fit

"""LIDAR CALIBRATION"""
# Load calibration data: column 0 = true distance, column 1 = measured distance
true_d, meas_d = loadtxt('lidar_cal.csv', skiprows=1, delimiter=',').T

def model(meas, a, b):
    return a + b * meas

def bresenham(r0, c0, r1, c1):
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    rr, cc = [], []
    r, c = r0, c0
    while True:
        rr.append(r); cc.append(c)
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc; r += sr
        if e2 < dr:
            err += dr; c += sc
    return np.array(rr), np.array(cc)

# Exclude readings at/near the max range where the sensor saturates
unfiltered = np.copy(meas_d)
unfiltered_true = np.copy(true_d)
valid = (meas_d > 0.15) & (meas_d < 6.0)
meas_d, true_d = meas_d[valid], true_d[valid]

# Initial fit on all data
params, cov = curve_fit(model, unfiltered, unfiltered_true)
true_fit = model(meas_d, *params)
residuals = true_d - true_fit

# Prune outliers beyond 2 std of residuals, refit
mask = abs(residuals) < std(residuals) * 2
pruned_meas = meas_d[mask]
pruned_true = true_d[mask]

params2, cov2 = curve_fit(model, pruned_meas, pruned_true)
pruned_fit = model(pruned_meas, *params2)
pruned_resid = pruned_true - pruned_fit

if 0: #Plotting
    print(f'Initial fit:  a={params[0]:.4f}, b={params[1]:.4f}')
    print(f'Pruned fit:   a={params2[0]:.4f}, b={params2[1]:.4f}')
    print(f'Outliers removed: {(~mask).sum()} of {len(meas_d)}')

    fig, axes = subplots(2)
    fig.subplots_adjust(hspace=0.5)
    axes[0].plot(unfiltered, unfiltered_true, '.', label='data')
    axes[0].plot(meas_d, true_fit, '.', label='fit')
    axes[0].set_xlabel('Measured (m)')
    axes[0].set_ylabel('True (m)')
    axes[0].set_title('LIDAR calibration')
    axes[0].legend()

    axes[1].set_title('Pruned LIDAR Calibration')
    axes[1].plot(pruned_meas, pruned_true, '.', label='pruned data')
    axes[1].plot(pruned_meas, pruned_fit, '.', label='fit')
    axes[1].set_xlabel('Measured (m)')
    axes[1].set_ylabel('True (m)')
    axes[1].legend()

    fig.savefig('lidar_calibration.png', dpi=300, bbox_inches='tight')
    show()

"""LIDAR TO WORLD FRAME"""
# lidar_scans.csv: robot_x, robot_y, robot_heading(deg), lidar_range(mm), lidar_angle(deg)
rx, ry, heading, r_mm, beam_ang = loadtxt('lidar_scans.csv', delimiter=',', skiprows=1).T

heading = np.deg2rad(heading)
beam_ang = np.deg2rad(beam_ang)

r_m = r_mm / 1000.0

# flag max-range / invalid returns before correcting
max_range_flag = r_m >= 6

r_corr = params2[0] + params2[1] * r_m

x_lidar = r_corr * np.cos(beam_ang)
y_lidar = r_corr * np.sin(beam_ang)

x_lidar_global = rx + x_lidar* np.cos(heading) - y_lidar * np.sin(heading)
y_lidar_global = ry + x_lidar* np.sin(heading) + y_lidar * np.cos(heading)

hit = ~max_range_flag
if 0: #Plot Cloud Plot
    fig1, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(12, 5))

    ax1a.scatter(x_lidar_global, y_lidar_global, s=2, label='LIDAR returns')
    ax1a.plot(rx, ry, 'r-', linewidth=2, label='robot path')
    ax1a.set_aspect('equal', adjustable='box')
    ax1a.legend()
    ax1a.set_xlabel('x (m)'); ax1a.set_ylabel('y (m)')
    ax1a.set_title('Unfiltered point cloud')

    ax1b.scatter(x_lidar_global[hit], y_lidar_global[hit], s=2, label='LIDAR returns')
    ax1b.plot(rx, ry, 'r-', linewidth=2, label='robot path')
    ax1b.set_aspect('equal', adjustable='box')
    ax1b.legend()
    ax1b.set_xlabel('x (m)'); ax1b.set_ylabel('y (m)')
    ax1b.set_title('Filtered point cloud')

    fig1.tight_layout()
    fig1.savefig('Cloud Plot.png', dpi=300, bbox_inches='tight')
    plt.show()

    fig2, ax2 = plt.subplots()
    ax2.plot(rx, ry, 'r-', linewidth=2, label='robot path')
    ax2.set_aspect('equal', adjustable='box')
    ax2.legend()
    ax2.set_xlabel('x (m)'); ax2.set_ylabel('y (m)')
    ax2.set_title('Robot path')
    fig2.savefig('Robot path.png', dpi=300, bbox_inches='tight')
    plt.show()

"""OCCUPANCY GRID"""
#Grid 201 x 51, grid size 50 x 50mm
cell_size = 0.05
n_rows, n_cols = 51, 201

y0 = 0
x0 = 0

def to_grid(x, y):
    row = np.round((y - y0) / cell_size).astype(int)
    col = np.round((x - x0) / cell_size).astype(int)
    return row, col

if 1:
    Phit, Pmiss = 0.9, 0.3
    l_hit = np.log(Phit/(1-Phit))
    l_miss = np.log(Pmiss/(1-Pmiss))
    logodds = np.zeros((n_rows, n_cols))

    r0_all, c0_all = to_grid(rx, ry)
    r1_all, c1_all = to_grid(x_lidar_global, y_lidar_global)

    for i in range(len(rx)):
        r0, c0 = r0_all[i], c0_all[i]
        r1, c1 = r1_all[i], c1_all[i]
        if not (0 <= r1 < n_rows and 0 <= c1 < n_cols):
            continue

        rr, cc = bresenham(r0, c0, r1, c1)
        valid_cells = (rr >= 0) & (rr < n_rows) & (cc >= 0) & (cc < n_cols)
        rr, cc = rr[valid_cells], cc[valid_cells]

        logodds[rr[:-1], cc[:-1]] += l_miss
        if not max_range_flag[i]:
            logodds[rr[-1], cc[-1]] += l_hit

    np.clip(logodds, -10, 10, out=logodds)
    prob = 1 - 1/(1 + np.exp(logodds))

    fig, ax = subplots(figsize=(12, 4))
    ax.imshow(prob, cmap='gray_r', origin='lower',
            extent=[x0, x0 + n_cols*cell_size, y0, y0 + n_rows*cell_size])
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_title('Occupancy grid')
    fig.savefig('Occupancy grid.png', dpi=300, bbox_inches='tight')
    show()