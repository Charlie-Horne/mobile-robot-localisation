import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt, arctan2, zeros, array, loadtxt, cos, sin

# Calibration from ankh.csv
a = -5.0021336
b = 0.20000534

# Beacon positions
xr1, yr1 = 0, 0
xr2, yr2 = 1000, 0

x =0
y=0
theta=0
v=0
X = np.array((x, y, theta, v))

t, x, y, z1, z2 = loadtxt('training.csv', skiprows=1, delimiter=',').T

def h(r, a, b):
    return (a + b * r) * (r >= 50)

# Direct distance between robot and beacon
r1 = sqrt((x - xr1)**2 + (y - yr1)**2)
r2 = sqrt((x - xr2)**2 + (y - yr2)**2)

# calibration to get h(x)
h1 = h(r1, a, b)
h2 = h(r2, a, b)

# Sensor noise/Residuals
v_a = z1 - h1
v_b = z2 - h2

print(f"beacon A: mean(v_a) = {v_a.mean():.5f}, std(v_a) = {v_a.std():.5f}, var(v_a) = {v_a.var():.5f}")
print(f"beacon B: mean(v_b) = {v_b.mean():.5f}, std(v_b) = {v_b.std():.5f}, var(v_b) = {v_b.var():.5f}")

plt.figure()
plt.plot(r1, v_a, '.')
plt.xlabel('$r_{a,n}$')
plt.ylabel('$v_{a,n}$')
plt.title('Beacon A residuals')

plt.figure()
plt.plot(r2, v_b, '.')
plt.xlabel('$r_{b,n}$')
plt.ylabel('$v_{b,n}$')
plt.title('Beacon B residuals')

plt.show()