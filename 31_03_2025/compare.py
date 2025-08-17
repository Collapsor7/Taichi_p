#
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os, sys, glob
# Число масс в цепочке
N = 10
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
dt = 1E-6
# Количество шагов по времени
nstep = int(t/dt)
# Массив времени
time = np.linspace(0,t,nstep)
# Массив положений масс
# ----------------- 0. 预分配并给初值 -----------------
x   = np.zeros((N, nstep+1), float)      # Euler  位置
vx  = np.zeros((N, nstep+1), float)      # Euler  速度
x_v = np.zeros((N, nstep+1), float)      # Verlet 位置
vx_v= np.zeros((N, nstep+1), float)      # Verlet 速度 (用中央差分算)

x[:,0]   = d*np.arange(N)/N + x0         # 初始直链
x_v[:,0] = x[:,0].copy()                 # 两方法同起点

l = d/N                                  # 每段自然长度

# -------- 1⃣ 先做 Verlet 的第一次半步: r(t+Δt) ----------
dx0  = np.diff(x_v[:,0]) - l
F0   = -C*np.append(0.0, dx0) + C*np.append(dx0, 0.0) - m*g
F0[0]-= Cs * x_v[0,0]*(x_v[0,0] < 0)     # 只在伸长时作用
a0_v  = F0/m
x_v[:,1] = x_v[:,0] + 0.5*a0_v*dt**2     # v₀=0，所以只有 ½aΔt²

# 可选：若你有非零初速度，把 vx_v[:,0] 赋上然后
#       x_v[:,1] = x_v[:,0] + vx_v[:,0]*dt + 0.5*a0_v*dt**2

# -------- 2⃣ 主循环：i 表示 “现在已知到 t = i·Δt” ----------
import time as pytime
tic = pytime.time()

for i in range(nstep):
    # ===== 欧拉 =======================================
    dx  = np.diff(x[:,i]) - l
    F   = -C*np.append(0.0, dx) + C*np.append(dx, 0.0) - m*g
    F[0]-= Cs * x[0,i]*(x[0,i] < 0)
    a    = F/m
    vx[:, i+1] = vx[:, i] + a*dt
    x [:, i+1] = x [:, i] + vx[:, i+1]*dt

    # ===== 位置‑Verlet (从 i>=1 开始才有 r_{i-1}) =====
    if i >= 1:
        dx_v  = np.diff(x_v[:, i]) - l
        F_v   = -C*np.append(0.0, dx_v) + C*np.append(dx_v, 0.0) - m*g
        F_v[0]-= Cs * x_v[0,i]*(x_v[0,i] < 0)
        a_v   = F_v / m
        x_v[:, i+1] = 2*x_v[:, i] - x_v[:, i-1] + a_v*dt**2          # r(t+Δt)

        # 用中央差分求 v(t)  → 公式中的第二行
        vx_v[:, i]   = (x_v[:, i+1] - x_v[:, i-1]) / (2*dt)

# 循环完后再补最后一格速度 v(t_final)
vx_v[:, -1] = (x_v[:, -1] - x_v[:, -2]) / dt

print(f"双方法计算完成，用时 {pytime.time()-tic:.1f} 秒")

fig = plt.figure(figsize=(20,20))

ax1 = fig.add_subplot(511)
ax1.set_title(u'Колебания масс', fontsize=24)
ax1.plot(time,x[4,0:-1], label = "Euler")
ax1.plot(time,x_v[4,0:-1], color='red',label= "Verlet")
ax1.set_xticks(np.round(np.linspace(0,t,51),decimals=2))
ax1.set_xticklabels(list(np.round(np.linspace(0,t,51),decimals=2)),fontsize=10,rotation=45)
ax2 = fig.add_subplot(512)
ax2.plot(time,x[3,0:-1])
ax2.plot(time,x_v[3,0:-1], color='red',label= "Verlet")
ax2.set_xticks(np.round(np.linspace(0,t,51),decimals=2))
ax2.set_xticklabels(list(np.round(np.linspace(0,t,51),decimals=2)),fontsize=10,rotation=45)
ax3 = fig.add_subplot(513)
ax3.plot(time,x[2,0:-1])
ax3.plot(time,x_v[2,0:-1], color='red',label= "Verlet")
ax3.set_xticks(np.round(np.linspace(0,t,51),decimals=2))
ax3.set_xticklabels(list(np.round(np.linspace(0,t,51),decimals=2)),fontsize=10,rotation=45)
ax4 = fig.add_subplot(514)
ax4.plot(time,x[1,0:-1])
ax4.plot(time,x_v[1,0:-1], color='red',label= "Verlet")
ax4.set_xticks(np.round(np.linspace(0,t,51),decimals=2))
ax4.set_xticklabels(list(np.round(np.linspace(0,t,51),decimals=2)),fontsize=10,rotation=45)
ax5 = fig.add_subplot(515)
ax5.plot(time,x[0,0:-1])
ax5.plot(time,x_v[0,0:-1], color='red',label= "Verlet")
ax5.set_xticks(np.round(np.linspace(0,t,51),decimals=2))
ax5.set_xticklabels(list(np.round(np.linspace(0,t,51),decimals=2)),fontsize=10,rotation=45)

for ax in fig.axes:
    ax.grid(True)

plt.tight_layout(h_pad = -2.5)

# plt.savefig(os.path.join(FPath,FName0+".png"), dpi=600, bbox_inches='tight')

plt.show()