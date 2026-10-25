from numpy import loadtxt
import numpy as np
from matplotlib.pyplot import subplots, show
from scipy.optimize import curve_fit

# Load calibration data
r, z = loadtxt('ankh.csv', skiprows=1, delimiter=',').T

# def model(r, a, b, c):
#     # residual
#     return a + b * r + c * r * r

# def model(r, a, b, c):
#     return a * (r < 50) + (b * r) * (r >= 50)

def model(r, a, b):

    return (a + b*r)  * (r >= 50) 


# def model(r, a, b):
#     return (a + b*r)  

# Load data r, z
params, cov = curve_fit(model, r, z)
zfit = model(r, *params)
residuals = z - zfit

print(params)
print(np.var(residuals))

# residuals = data - model
# mask = abs(residuals) < np.std(residuals) * 5
# pruned_data = data[mask]


fig, axes = subplots(2)
axes[0].plot(r, z, '.')
axes[0].plot(r, zfit, '.')
axes[0].set_ylabel('Measurement')
axes[0].set_title('Ankh calibration data')


axes[1].plot(r, residuals, '.')
axes[1].set_xlabel('Distance (m)')
axes[1].set_ylabel('Measurement')
axes[1].set_title('Ankh calibration data')

show()


