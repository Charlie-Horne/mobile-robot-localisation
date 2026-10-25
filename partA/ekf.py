from numpy.linalg import inv


class EKF:
    """Extended Kalman filter"""

    def __init__(self, xv0, P0):
        """
        Initialise EKF

        Parameters
        ----------
        `xv0` mean vector of initial state
        `P0`  covariance matrix of initial state

        Usage
        -----
        ekf = EKF(xv0, P0)
        ekf.predict(uv, A, gv_func, Q)
        ekf.update(zv, C, hv_func, R)

        xv = ekf.xv  # Mean vector
        P = ekf.P    # Covariance matrix
        """

        self.xv = xv0
        self.P = P0
        self.K = None

    def predict(self, uv, A, gv_func, Q):
        """
        Perform predict step for EKF.

        Parameters
        ----------
        `uv` control vector
        `A`  state transition matrix (for non-linear motion models
             this is the Jacobian of the motion model evaluated at the
             current estimate)
        `gv_func(xv, uv)` motion model function
        `Q`  covariance matrix of process noise vector, cov(W)

        Returns
        -------
        `xv`  Prior mean vector of state
        `P`   Prior covariance matrix of state

        

        """

        xv = self.xv
        P = self.P

        # Prior mean vector of state
        xvp = gv_func(xv, uv)

        # Prior covariance matrix of state
        Pp = A.dot(P).dot(A.T) + Q

        self.xv = xvp
        self.P = Pp

        return xvp, Pp

    def update(self, zv, C, hv_func, R):
        """
        Perform update step of EKF.

        Parameters
        ----------
        `zv` measurement vector
        `C`  sensor matrix (for non-linear sensor models this is
             the Jacobian of the sensor model evaluated at the
             current estimate)
        `hv_func(xv)` sensor model function
        `R`  covariance matrix of measurement noise vector

        Returns
        -------
        `xv`  Prior mean vector of state
        `P`   Prior covariance matrix of state
        """

        xvp = self.xv
        Pp = self.P

        # Innovation
        yv = zv - hv_func(xvp)

        # Innovation covariance
        S = C.dot(Pp).dot(C.T) + R

        # Kalman gain
        K = Pp.dot(C.T).dot(inv(S))

        # Posterior mean vector of state
        xv = xvp + K.dot(yv)

        # Posterior covariance matrix of state
        P = Pp - K.dot(C).dot(Pp)

        self.xv = xv
        self.P = P
        self.K = K

        return xv, P
