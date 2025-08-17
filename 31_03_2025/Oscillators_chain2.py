import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os, sys, glob
import time as pytime
import taichi as ti

ti.init(arch=ti.gpu)

N = 90000
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


l = d/N
print("numMassPoints:",N)
print("numSteps:",nstep)

time = ti.field(dtype=ti.f32, shape=(nstep+1))
F = ti.field(dtype=ti.f32, shape=N)
dx = ti.field(dtype=ti.f32, shape=N-1)

x1 = ti.field(dtype=float, shape=N)
x2 = ti.field(dtype=float, shape=N)
v1 = ti.field(dtype=float, shape=N)
v2 = ti.field(dtype=float, shape=N)

chunk_size = 50000
# x_chunk_ti = ti.field(dtype=ti.f32, shape=(N, chunk_size))
x_sum = np.zeros((N,nstep+1),float)

@ti.kernel
def Init():

    for i in range(N):
        x1[i] = (d/N) * i + x0
        v1[i] = 0
        x2[i] = (d/N) * i + x0
        v2[i] = 0
    for i in range(nstep + 1):
        time[i] = i * dt

Init()
x_sum[:, 0] = x1.to_numpy()

@ti.kernel
def substep():
    for j in range(N - 1):
        dx[j] = x1[j + 1] - x1[j] - l

    for j in range(N):
        left = -C * dx[j - 1] if j > 0 else 0.0
        right = C * dx[j] if j < N - 1 else 0.0

        contact = -Cs * x1[j] if j == 0 and x1[j] < 0.0 else 0.0

        F[j] = left + right - B * v1[j] + contact - m * g

    for j in range(N):
        a = F[j] / m
        v2[j] = v1[j] + a * dt
        x2[j] = x1[j] + v2[j] * dt

    for j in range(N):
        x1[j] = x2[j]
        v1[j] = v2[j]


start_time = pytime.time()
for i in range(nstep):
    substep()
    # p = x1.to_numpy()
    # x_sum[:,i+1] = p
print("Elapsed time:", pytime.time() - start_time)
#
# print(x_sum)
# print(x_sum.shape)

# yy= ti.field(dtype=ti.f32, shape=(nstep + 1))
# @ti.func
# def min():
#     yy = x[0]-vx[0]
#
# print(yy.to_numpy())
#
#
# print("dx",dx)
# print("f",F)
# xx=x.to_numpy()
# print("x",xx)
# print("vx",vx.to_numpy())
# time=time.to_numpy()
# print("time",time)

# step = int(x.shape[1]/1000)
# for j in range(0,int(x.shape[1])+1,step):
# #for j in range(0,1000+1,1):
#     plt.figure(figsize=(2,6))
#     plt.ylim(x[0,:].min(),x[-1,:].max())
#     # print(f'STEP: {j}')
#     #plt.title(f'STEP: {j}',fontsize=16)
#     plt.title(f'Время: {j*dt:.2e}',fontsize=16)
#     plt.scatter(np.ones(x.shape[0]),x[:,j])
#     # plt.savefig(os.path.join(FPath,"%020d.png" % j), dpi=300, bbox_inches='tight')
#     plt.close()
#     #sys.exit()

def drawPicture(xx):
    fig = plt.figure(figsize=(20,20))

    ax1 = fig.add_subplot(511)
    ax1.set_title(u'Колебания масс', fontsize=24)
    ax1.plot(time,xx[4,0:])
    ax1.set_xticks(np.round(np.linspace(0,t,51),decimals=2))
    ax1.set_xticklabels(list(np.round(np.linspace(0,t,51),decimals=2)),fontsize=10,rotation=45)
    ax2 = fig.add_subplot(512)
    ax2.plot(time,xx[3,0:])
    ax2.set_xticks(np.round(np.linspace(0,t,51),decimals=2))
    ax2.set_xticklabels(list(np.round(np.linspace(0,t,51),decimals=2)),fontsize=10,rotation=45)
    ax3 = fig.add_subplot(513)
    ax3.plot(time,xx[2,0:])
    ax3.set_xticks(np.round(np.linspace(0,t,51),decimals=2))
    ax3.set_xticklabels(list(np.round(np.linspace(0,t,51),decimals=2)),fontsize=10,rotation=45)
    ax4 = fig.add_subplot(514)
    ax4.plot(time,xx[1,0:])
    ax4.set_xticks(np.round(np.linspace(0,t,51),decimals=2))
    ax4.set_xticklabels(list(np.round(np.linspace(0,t,51),decimals=2)),fontsize=10,rotation=45)
    ax5 = fig.add_subplot(515)
    ax5.plot(time,xx[0,0:])
    ax5.set_xticks(np.round(np.linspace(0,t,51),decimals=2))
    ax5.set_xticklabels(list(np.round(np.linspace(0,t,51),decimals=2)),fontsize=10,rotation=45)

    for ax in fig.axes:
        ax.grid(True)

    plt.tight_layout(h_pad = -2.5)

    # plt.savefig(os.path.join(FPath,FName0+".png"), dpi=600, bbox_inches='tight')

    plt.show()

# drawPicture(x_sum)