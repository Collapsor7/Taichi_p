import numpy as np
import matplotlib.pyplot as plt

# ------------------------
# 1. 物理与数值参数
# ------------------------
N        = 5       # 质点数
m        = 0.1      # 质量 (kg)
C        = 100.0    # 相邻弹簧劲度
Cs       = 1000.0   # 顶端与“天花板”弹簧劲度
d        = 1.25     # 自然长度 (m)
g        = 9.81     # 重力加速度
x0       = 0.5      # 顶端质点静止时的位置 (m)
B        = 0.0      # 阻尼(此处忽略)
dt       = 1e-4     # 时间步长 (s) — 默认 10⁻⁴，内存/速度友好
t_total  = 3.0      # 模拟总时长 (s)
nstep    = int(t_total / dt)
time     = np.linspace(0.0, t_total, nstep)

# ------------------------
# 2. 辅助函数：计算加速度
# ------------------------
def acceleration(x):
    """给定当前所有质点的位置 x (shape=(N,)), 返回加速度数组 a"""
    a = np.empty_like(x)
    for i in range(N):
        F = -m * g                     # 重力
        if i > 0:                      # 上端弹簧
            F += C * ((x[i-1] - x[i]) - d)
        if i < N-1:                    # 下端弹簧
            F += -C * ((x[i] - x[i+1]) - d)
        if i == 0:                     # 顶端连接“天花板”的弹簧
            F += -Cs * (x[i] - x0)
        a[i] = F / m
    return a

# ------------------------
# 3 A. Euler 积分
# ------------------------
x_euler = np.zeros((nstep, N))
v_euler = np.zeros((nstep, N))
x_euler[0] = np.linspace(x0, x0 - (N-1)*d, N)         # 初始链条呈拉直状态

for k in range(nstep-1):
    a = acceleration(x_euler[k])
    v_euler[k+1] = v_euler[k] + a*dt
    x_euler[k+1] = x_euler[k] + v_euler[k+1]*dt

# ------------------------
# 3 B. Verlet (velocity‑Verlet) 积分
# ------------------------
x_verlet = np.zeros_like(x_euler)
v_verlet = np.zeros_like(v_euler)
x_verlet[0] = x_euler[0].copy()                       # 与 Euler 同起点
a_prev = acceleration(x_verlet[0])

for k in range(nstep-1):
    # 位置半步
    x_half = x_verlet[k] + v_verlet[k]*dt + 0.5*a_prev*dt**2
    # 下一步加速度
    a_next = acceleration(x_half)
    # 完成速度更新
    v_next = v_verlet[k] + 0.5*(a_prev + a_next)*dt
    # 写入
    x_verlet[k+1] = x_half
    v_verlet[k+1] = v_next
    a_prev = a_next

# ------------------------
# 4. 绘图：每条子图对应一个质点
# ------------------------
fig, axes = plt.subplots(N, 1, sharex=True,
                         figsize=(12, 2.2*N), dpi=110)

for i, ax in enumerate(axes):
    ax.plot(time, x_euler[:, i], lw=1.0, label='Euler')
    ax.plot(time, x_verlet[:, i], lw=1.0, ls='--', color='orangered',
            label='Verlet')
    ax.set_ylabel(f'mass {i}')
    ax.grid(True, which='both', ls=':', lw=0.5)
    if i == 0:               # 只在第一幅加图例
        ax.legend(loc='best')

axes[-1].set_xlabel('Time (s)')
fig.suptitle('Колебания масс', fontsize=16, y=0.92)
plt.tight_layout()
plt.show()