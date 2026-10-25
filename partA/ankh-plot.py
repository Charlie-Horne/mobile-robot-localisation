from numpy import loadtxt
from matplotlib.pyplot import subplots, show

# Load calibration data
r, z = loadtxt('ankh.csv', skiprows=1, delimiter=',').T

fig, axes = subplots(1)
axes.plot(r, z, '.')
axes.set_xlabel('Distance (m)')
axes.set_ylabel('Measurement')
axes.set_title('Ankh calibration data')

show()
