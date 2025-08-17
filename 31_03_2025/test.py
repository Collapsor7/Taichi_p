import taichi as ti
import numpy as np

# ti.init(arch=ti.gpu)
#
# x1 = ti.field(int,shape=5)
# x2 = ti.field(int,shape=5)
#
#
# @ti.kernel
# def test():
#     x1 = ti.Vector([2,3,4,5,1])
#
# test()
# print(x1.to_numpy())
# n= 5
# B = 0
# C = 100
# Cs = 1000
# d = 1.25
# m = 0.1
# t = 3.0
# g = 9.81
# x0 = 0.5
# dt = 1E-5
# nstep = int(t/dt)
#
# x1 = np.zeros((n, 3), dtype=np.float32)
# print("x1",x1)
# for i in range(n):
#     x1[i] = [0,(d/n) * i + x0,0]
# print("x1",x1)

x = np.zeros((2, 3), dtype=np.float32)
print(x)