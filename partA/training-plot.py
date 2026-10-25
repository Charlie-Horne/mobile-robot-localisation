"""
Find W for motion model
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt, arctan2, zeros, array, loadtxt, cos, sin


t, x, y, z1, z2 = loadtxt('training.csv', skiprows=1, delimiter=',').T

x0 = x[0]
y0 = y[0]

dt = t[1] - t[0]

vx = (np.mean(x))/dt
vy = (np.mean(y))/dt

theta = arctan2(vy, vx)
theta_deg = theta * (180/np.pi)

F = array([[1/2 * dt**2 * cos(theta)], 
          [1/2 * dt**2 * sin(theta)],
          [dt*cos(theta)],
          [dt*sin(theta)]])


s = sqrt((x-x0)**2 + (y-y0)**2)

v = np.gradient(s,t)
a = np.gradient(v, t)

plt.figure()
# plt.plot(t,s,'.')
# plt.plot(t,v,'.')
# plt.plot(t,a,'.')
# plt.xlabel('$t$')
# plt.ylabel('$s$')
plt.plot(x, y, '.')

# plt.plot(t, theta, '.')

# print(np.mean(a))
print(np.var(a))
print(theta)
print(theta_deg)

# print(np.size(F))

Q = F @ F.T * np.var(a)

print(Q)
# plt.show()