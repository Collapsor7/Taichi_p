import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os, sys, glob
import time


n = 100
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
# time = np.linspace(0,t,nstep)
# x = np.zeros((n,nstep+1),float)
# vx = np.zeros((N,nstep+1),float)
#
# x[:,0] = d*(np.array(range(N)))/N+x0
l = d/n

# print("numMassPoints:",N)
# print("numSteps:",nstep)
#
# import time as pytime
# start = pytime.time()
# for i in range(nstep):
#     dx = np.diff(x[:,i])-l
#     F = -C*np.append(0.0,dx) + C*np.append(dx,0.0) - m*g
#     F[0] = F[0] - Cs*x[0,i]*(x[0,i]<0)
#     a = F/m
#     vx[:,i+1] = vx[:,i] + a*dt
#     x[:,i+1] = x[:,i] + vx[:,i+1]*dt
# print(pytime.time()-start)
# print(x)

# x_sum = []
# v_sum = []
#
# x1 = np.zeros(N,float)
# x2 = np.zeros(N,float)
#
# v1 = np.zeros(N,float)
# v2 = np.zeros(N,float)
#
# x1 = d*(np.array(range(N)))/N+x0
# print("x1",x1)
# x_sum.append(x1.tolist())
# print("x_sum",x_sum)
# for i in range(nstep):
#     dx = np.diff(x1) - l
#     F = -C*np.append(0.0,dx) + C*np.append(dx,0.0) - m*g
#     F[0] = F[0] - Cs * x1[0] * (x1[0] < 0)
#     a = F/m
#     v2 = v1 + a * dt
#     x2 = x1 + v2 * dt
#     x1, v1 = x2, v2
#     x_sum.append(x1.tolist())
#     v_sum.append(v1.tolist())
#
# print(x_sum)

x = np.zeros((n, 3), dtype=np.float32)
v = np.zeros((n, 3), dtype=np.float32)


dx = np.zeros((n - 1, 3), dtype=np.float32)
F = np.zeros((n, 3), dtype=np.float32)
a = np.zeros(3, dtype=np.float32)

l = d / n
for i in range(n):
    x[i] = [0, (d / n) * i + x0, 0]

    # Замер времени выполнения
start_time = time.time()

for i in range(nstep):
    for j in range(n - 1):
        dx[j] = x[j + 1] - x[j] - l

    for j in range(n):

        left = np.array([0.0, 0.0, 0.0])
        right = np.array([0.0, 0.0, 0.0])
        contact = np.array([0.0, 0.0, 0.0])

        if j > 0:
            left = -C * dx[j - 1]
        if j < n - 1:
            right = C * dx[j]
        if j == 0 and x[j][1] < 0.0:
            contact = -Cs * x[j]

        F[j] = left + right - B * v[j] + contact - m * np.array([0, 9.8, 0])

    for j in range(n):
        a = F[j] / m
        v[j] = v[j] + a * dt
        x[j] = x[j] + v[j] * dt

end_time = time.time()
print(end_time - start_time)