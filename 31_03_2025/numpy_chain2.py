import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os, sys, glob


N = 5
B = 0
C = 100
Cs = 1000
d = 1.25
m = 0.1
t = 3.0
g = 9.81
x0 = 0.5
dt = 1E-5

nstep = int(t/dt)
time = np.linspace(0,t,nstep)
x = np.zeros((N,nstep+1),float)
vx = np.zeros((N,nstep+1),float)

x[:,0] = d*(np.array(range(N)))/N+x0
l = d/N

print("numMassPoints:",N)
print("numSteps:",nstep)
#
# import time as pytime
# start = pytime.time()
# for i in range(nstep):
#     # print(f'STEP: {i}')
#     dx = np.diff(x[:,i])-l
#     F = -C*np.append(0.0,dx) + C*np.append(dx,0.0) - m*g
#     F[0] = F[0] - Cs*x[0,i]*(x[0,i]<0)
#     a = F/m
#     vx[:,i+1] = vx[:,i] + a*dt
#     x[:,i+1] = x[:,i] + vx[:,i+1]*dt
# print(pytime.time()-start)
# print(x.shape)

x_sum = []
v_sum = []

x1 = np.zeros(N,float)
x2 = np.zeros(N,float)

v1 = np.zeros(N,float)
v2 = np.zeros(N,float)

x1 = d*(np.array(range(N)))/N+x0
print("x1",x1)
x_sum.append(x1.tolist())
# print("x_sum",x_sum)

import time as pytime
start = pytime.time()
for i in range(nstep):
    dx = np.diff(x1) - l
    F = -C*np.append(0.0,dx) + C*np.append(dx,0.0) - m*g
    F[0] = F[0] - Cs * x1[0] * (x1[0] < 0)
    a = F/m
    v2 = v1 + a * dt
    x2 = x1 + v2 * dt
    x1, v1 = x2, v2

#     x_sum.append(x1.tolist())
#     v_sum.append(v1.tolist())
#
# print(x_sum)

print(pytime.time()-start)
