import numpy as np
from numpy import sqrt, arctan2, zeros, array, loadtxt, cos, sin
from ekf import EKF

t, x, y, z1, z2 = loadtxt('training.csv', skiprows=1, delimiter=',').T

# Beacon positions
xr1, yr1 = 0, 0
xr2, yr2 = 1000, 0

dt = t[1] - t[0]

# Sensor model values from ankh.csv calibration
a = -5.0021336
b = 0.20000534

accl_variance = 0.0023373027664852068
theta = 0.5991760660776833

def h(r):
    #Sensor model
    return (a + b * r) * (r >= 50)

# TODO: Define initial mean of state vector xv0

xv0 = array([x[0], y[0], (x[1] - x[0]) / dt, (y[1] - y[0]) / dt])

# TODO: Define initial covariance of state vector P0
P0 = array([[0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0],
           [0, 0, 0, 0]])

# TODO: Define motion model matrix A
A = array([[1, 0, dt, 0],
          [0, 1, 0, dt],
          [0, 0, 1, 0],
          [0, 0, 0, 1]])


def gv_func(xv, uv):
    """
    Motion model

    Parameters
    ----------
    `xv` state vector
    `uv` control vector

    Returns
    -------
    `xvp` predicted state vector

    """
    # TODO: Return predicted state vector using motion model

    xvp = A @ xv
     
    return xvp


def hv_func(xv):
    """
    Sensor model in terms of state vector

    Parameters
    ----------
    `xv` state vector

    Returns
    -------
    `zv` predicted measurement vector

    """
    # TODO: Calculate z1 and z2 using sensor model and beacon positions

    r1 = sqrt( (xv[0] - xr1)**2 +  (xv[1] - yr1)**2)
    r2 = sqrt( (xv[0] - xr2)**2 +  (xv[1] - yr2)**2)

    z1 = h(r1)
    z2 = h(r2)

    return array((z1, z2))


def C_func(xv):
    """
    Sensor model Jacobian evaluated at state estimate

    Parameters
    ----------
    `xv` state vector estimate to linearise about

    Returns
    -------
    `C` Jacobian matrix evaluated at state estimate

    """

    # TODO: Calculate Jacobian matrix to linearise sensor model and
    # evaluate at xv to find C.

    # z1 = br + a

    r1 = sqrt( (xv[0] - xr1)**2 +  (xv[1] - yr1)**2)
    r2 = sqrt( (xv[0] - xr2)**2 +  (xv[1] - yr2)**2)

    # dz1dx = b*(x[0]/r1)
    # dz1dy = b*(x[1]/r1)
    # dz2dx = b*(x[0]/r2)
    # dz2dy = b*(x[1]/r2)

    J = array([[ b*(xv[0]/r1), b*(xv[1]/r1), 0, 0],
              [ b*(xv[0]/r2), b*(xv[1]/r2), 0, 0]])

    C = J

    return C


N = len(t)

# Measurements
measurements = array((z1, z2))

# There is no control input
uv = 0

ekf = EKF(xv0, P0)

for n in range(N):

    # TODO: Determine process noise covariance matrix Q
    F = array([[1/2 * dt**2 * cos(theta)], 
          [1/2 * dt**2 * sin(theta)],
          [dt*cos(theta)],
          [dt*sin(theta)]])
    
    Q = F @ F.T * accl_variance

    ekf.predict(uv, A, gv_func, Q)

    zv = measurements[:, n]
    # TODO: Handle measurement outliers
    

    # TODO: Determine sensor noise covariance matrix R
    R = array([[0.0347138, 0],
            [0, 0.0347138]])


    C = C_func(ekf.xv)
    ekf.update(zv, C, hv_func, R)

    # TODO: Save history of state mean and covariance for plotting

# TODO: Plot results
