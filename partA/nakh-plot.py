from numpy import loadtxt
import numpy as np
from matplotlib.pyplot import subplots, show
from scipy.optimize import curve_fit

# Load calibration data
r, z = loadtxt('nakh.csv', skiprows=1, delimiter=',').T


def model(r, a, b):
    return (a + b*r)  


params, cov = curve_fit(model, r, z)
zfit = model(r, *params)
residuals = z - zfit

data = z
# residuals = data - model
mask = abs(residuals) < np.std(residuals) * 2
pruned_z = z[mask]
pruned_r = r[mask]

params2, cov2 = curve_fit(model, pruned_r, pruned_z)
pruned_zfit = model(pruned_r, *params2)
pruned_resid = pruned_z - pruned_zfit

fig, axes = subplots(2)
axes[0].plot(r, z, '.')
axes[0].plot(r, zfit, '.')
axes[0].set_ylabel('Measurement')
axes[0].set_xlabel('Range')
axes[0].set_title('Nakh')


axes[1].plot(pruned_r, pruned_z, '.', label='pruned data')
axes[1].plot(pruned_r, pruned_zfit, '.', label='fit')
# axes[1].plot(r, residuals, '.')
axes[1].set_xlabel('Distance (m)')
axes[1].set_ylabel('Measurement')


show()


