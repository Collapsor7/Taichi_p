import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os, sys, glob
import time as pytime
import taichi as ti

ti.init(arch=ti.gpu)

# Число масс в цепочке
N = 5
# Константа демпфирования
B = 0
# Жесткость пружин
C = 100
# Жесткость основания
Cs = 1000
# Размер цепочки в м
d = 1.25
# Массы в кг
m = 0.1
# Время моделирования
t = 3.0
# Ускорение свободного падения
g = 9.81
# Высота падения
x0 = 0.5
# Шаг интегрирования по времени
dt = 1E-5
# dt = 0.1
# Количество шагов по времени
nstep = int(t/dt)
# Массив времени
# time = np.linspace(0,t,nstep)
time = ti.field(dtype=ti.f32, shape=(nstep+1))
# Массив положений масс
# x = np.zeros((N,nstep+1),float)
x = ti.field(dtype=float, shape=(N, nstep + 1))
# Массив скоростей масс
# vx = np.zeros((N,nstep+1),float)
vx = ti.field(dtype=float, shape=(N, nstep + 1))
# Начальные положения масс
# x[:,0] = d*(np.array(range(N)))/N+x0
# Расновесные расстояния между массами, т.е. пружины не растянуты и не сжаты
l = d/N
print("numMassPoints:",N)
print("numSteps:",nstep)

F = ti.field(dtype=ti.f32, shape=N)
dx = ti.field(dtype=ti.f32, shape=N-1)

@ti.kernel
def Init():
    for i,j in ti.ndrange(N,nstep+1):
        x[i,j] = (d/N) * i + x0 if j == 0 else 0.0
        vx[i,j] = 0
        time[j] = j * dt

Init()
# print(x.to_numpy())


@ti.kernel
def substep(i:int):
    for j in range(N - 1):
        dx[j] = (x[j + 1, i] - x[j, i]) - l

    for j in range(N):
        left = -C * dx[j - 1] if j > 0 else 0.0
        right = C * dx[j] if j < N - 1 else 0.0

        contact = -Cs * x[j, i] if j == 0 and x[j, i] < 0.0 else 0.0

        F[j] = left + right - B * vx[j, i] + contact - m * g

    for j in range(N):
        a = F[j] / m
        vx[j, i + 1] = vx[j, i] + a * dt
        x[j, i + 1] = x[j, i] + vx[j, i + 1] * dt


start_time = pytime.time()
for i in range(nstep):
    substep(i)
print("Elapsed time:", pytime.time() - start_time)
print(x)

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

drawPicture(x.to_numpy())
print(x.to_numpy().shape)