import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os, sys, glob
import time
import taichi as ti

ti.init(arch=ti.gpu)

n = 10
B = 0
C = 100
Cs = 1000
d = 1.25
m = 0.1
t = 3.0
g = 9.81
x0 = 0.5
dt = 1E-5
# dt = 0.1

nstep = int(t/dt)


l = d/n
print("numMassPoints:",n)
print("numSteps:",nstep)
block_size = 8
grid_size = (n + block_size - 1) // block_size
print("grid_size:",grid_size)

x = ti.Vector.field(3, dtype=ti.f32)
v = ti.Vector.field(3, dtype=ti.f32)
ti.root.dense(ti.i, grid_size).dense(ti.i, block_size).place(x)
ti.root.dense(ti.i, grid_size).dense(ti.i, block_size).place(v)

dx = ti.Vector.field(3, dtype=ti.f32)
ti.root.dense(ti.i, grid_size).dense(ti.i, block_size).place(dx)
F = ti.Vector.field(3, dtype=ti.f32)
ti.root.dense(ti.i, grid_size).dense(ti.i, block_size).place(F)


@ti.kernel
def Init():
    for i in range(n):
        x[i] = [0, l * i + x0, 0]
        v[i] = [0, 0, 0]

@ti.kernel
def substep():
    for j in range(n - 1):
        dx[j] = x[j + 1] - x[j] -ti.Vector([0, l,0])

    for j in range(n):
        left = -C * dx[j - 1] if j > 0 else 0.0
        right = C * dx[j] if j < n - 1 else 0.0
        contact = -Cs * x[j] if j == 0 and x[j][1] < 0.0 else 0.0
        F[j] = left + right - B * v[j] + contact - m * ti.Vector([0, 9.8, 0])

    for j in range(n):
        a = F[j] / m
        v[j] = v[j] + a * dt
        x[j] = x[j] + v[j] * dt

Init()
print(x.to_numpy())

# Замер времени выполнения
start_time = time.time()
for i in range(nstep):
    substep()
end_time = time.time()
print("Elapsed time:", end_time - start_time)
print(x.to_numpy())
