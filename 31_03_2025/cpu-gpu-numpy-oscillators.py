import taichi as ti
import time
import numpy as np

B = 0
C = 100
Cs = 1000
d = 1.25
m = 0.1
t = 3.0
g = 9.81
x0 = 0.5
dt = 1E-5
nstep = int(t / dt)


def compute_taichi_gpu(n):
    ti.init(arch=ti.gpu)

    x = ti.Vector.field(3, dtype=ti.f32, shape=n)
    v = ti.Vector.field(3, dtype=ti.f32, shape=n)

    dx = ti.Vector.field(3, dtype=ti.f32, shape=(n - 1))
    F = ti.Vector.field(3, dtype=ti.f32, shape=n)

    l = d / n

    @ti.kernel
    def Init():
        for i in range(n):
            x[i] = [0, l * i + x0, 0]
            v[i] = [0, 0, 0]

    @ti.kernel
    def substep():
        for j in range(n - 1):
            dx[j] = x[j + 1] - x[j] - ti.Vector([0, l, 0])

        for j in range(n):

            left = ti.Vector([0.0, 0.0, 0.0])
            right = ti.Vector([0.0, 0.0, 0.0])
            contact = ti.Vector([0.0, 0.0, 0.0])

            if j > 0:
                left = -C * dx[j - 1]
            if j < n - 1:
                right = C * dx[j]
            if j == 0 and x[j][1] < 0.0:
                contact = -Cs * x[j]

            F[j] = left + right - B * v[j] + contact - m * ti.Vector([0, 9.8, 0])

        for j in range(n):
            a = F[j] / m
            v[j] = v[j] + a * dt
            x[j] = x[j] + v[j] * dt

    # # Заполнение массива случайными значениями
    Init()

    # Замер времени выполнения
    start_time = time.time()
    for i in range(nstep):
        substep()
    end_time = time.time()

    return end_time - start_time

def compute_taichi_threads_blocks_managment(n, cpu_max_num_threads, block_size):
    ti.reset()
    ti.init(arch=ti.cpu, cpu_max_num_threads=cpu_max_num_threads)

    # Организация данных через плотные массивы с заданным размером блока
    grid_size = (n + block_size - 1) // block_size  # Вычисление размера сетки

    print(grid_size)
    x = ti.Vector.field(3, dtype=ti.f32)
    v = ti.Vector.field(3, dtype=ti.f32)

    ti.root.dense(ti.i, grid_size).dense(ti.i, block_size).place(x)
    ti.root.dense(ti.i, grid_size).dense(ti.i, block_size).place(v)

    dx = ti.Vector.field(3, dtype=ti.f32)
    ti.root.dense(ti.i, grid_size).dense(ti.i, block_size).place(dx)
    F = ti.Vector.field(3, dtype=ti.f32)
    ti.root.dense(ti.i, grid_size).dense(ti.i, block_size).place(F)


    l = d / n

    @ti.kernel
    def Init():
        for i in range(n):
            x[i] = [0, l * i + x0, 0]
            v[i] = [0, 0, 0]

    @ti.kernel
    def substep():
        for j in range(n - 1):
            dx[j] = x[j + 1] - x[j] - ti.Vector([0, l, 0])

        for j in range(n):

            left = ti.Vector([0.0, 0.0, 0.0])
            right = ti.Vector([0.0, 0.0, 0.0])
            contact = ti.Vector([0.0, 0.0, 0.0])

            if j > 0:
                left = -C * dx[j - 1]
            if j < n - 1:
                right = C * dx[j]
            if j == 0 and x[j][1] < 0.0:
                contact = -Cs * x[j]

            F[j] = left + right - B * v[j] + contact - m * ti.Vector([0, 9.8, 0])

        for j in range(n):
            a = F[j] / m
            v[j] = v[j] + a * dt
            x[j] = x[j] + v[j] * dt

    # # Заполнение массива случайными значениями
    Init()

    # Замер времени выполнения
    start_time = time.time()
    for i in range(nstep):
        substep()
    end_time = time.time()

    return end_time - start_time

#
# Функция для вычисления сил в NumPy
def compute_numpy(n):
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

    return end_time - start_time
#
#
# # Функция для вычисления сил в Taichi
def compute_taichi_memory_management(n, cpu_max_num_threads, memory_layout):
    ti.reset()
    ti.init(arch=ti.cpu, cpu_max_num_threads=cpu_max_num_threads)

    l = d / n

    if memory_layout == "dense":
        # Плотные массивы
        block_size = 16
        grid_size = (n + block_size - 1) // block_size  # Вычисление размера сетки
        x = ti.Vector.field(3, dtype=ti.f32)
        v = ti.Vector.field(3, dtype=ti.f32)
        ti.root.dense(ti.i, grid_size).dense(ti.i, block_size).place(x)
        ti.root.dense(ti.i, grid_size).dense(ti.i, block_size).place(v)

        dx = ti.Vector.field(3, dtype=ti.f32)
        ti.root.dense(ti.i, grid_size).dense(ti.i, block_size).place(dx)
        F = ti.Vector.field(3, dtype=ti.f32)
        ti.root.dense(ti.i, grid_size).dense(ti.i, block_size).place(F)

    elif memory_layout == "pointer":
        # Разреженные массивы
        block_size = 16
        grid_size = (n + block_size - 1) // block_size  # Вычисление размера сетки
        x = ti.Vector.field(3, dtype=ti.f32)
        v = ti.Vector.field(3, dtype=ti.f32)

        ti.root.pointer(ti.i, grid_size).dense(ti.i, block_size).place(x)
        ti.root.pointer(ti.i, grid_size).dense(ti.i, block_size).place(v)

        dx = ti.Vector.field(3, dtype=ti.f32)
        ti.root.pointer(ti.i, grid_size).dense(ti.i, block_size).place(dx)
        F = ti.Vector.field(3, dtype=ti.f32)
        ti.root.pointer(ti.i, grid_size).dense(ti.i, block_size).place(F)

    elif memory_layout == "vector":
        # Использование ti.Vector
        x = ti.Vector.field(3, dtype=ti.f32, shape=n)
        v = ti.Vector.field(3, dtype=ti.f32, shape=n)

        dx = ti.Vector.field(3, dtype=ti.f32, shape=(n-1))
        F = ti.Vector.field(3, dtype=ti.f32, shape=n)

    @ti.kernel
    def Init():
        for i in range(n):
            x[i] = [0, l * i + x0, 0]
            v[i] = [0, 0, 0]

    @ti.kernel
    def substep():
        for j in range(n - 1):
            dx[j] = x[j + 1] - x[j] - ti.Vector([0, l, 0])

        for j in range(n):

            left = ti.Vector([0.0, 0.0, 0.0])
            right = ti.Vector([0.0, 0.0, 0.0])
            contact = ti.Vector([0.0, 0.0, 0.0])

            if j > 0:
                left = -C * dx[j - 1]
            if j < n - 1:
                right = C * dx[j]
            if j == 0 and x[j][1] < 0.0:
                contact = -Cs * x[j]

            F[j] = left + right - B * v[j] + contact - m * ti.Vector([0, 9.8, 0])

        for j in range(n):
            a = F[j] / m
            v[j] = v[j] + a * dt
            x[j] = x[j] + v[j] * dt

    # # Заполнение массива случайными значениями
    Init()

    # Замер времени выполнения
    start_time = time.time()
    for i in range(nstep):
        substep()
    end_time = time.time()

    return end_time - start_time


results={}
threads_blocks_managment_settings = [
    {"cpu_max_num_threads": 1, "block_size": 1, "label": "Threads=1, Block=1"},
    {"cpu_max_num_threads": 4, "block_size": 8, "label": "Threads=4, Block=8"},
    {"cpu_max_num_threads": 8, "block_size": 16, "label": "Threads=8, Block=16"},
    {"cpu_max_num_threads": 8, "block_size": 32, "label": "Threads=8, Block=32"},
    {"cpu_max_num_threads": 16, "block_size": 64, "label": "Threads=16, Block=64"},
]
memory_management_settings = [
    {"cpu_max_num_threads": 8, "memory_layout": "dense", "label": "Dense Memory"},
    {"cpu_max_num_threads": 8, "memory_layout": "pointer", "label": "Pointer Memory"},
    {"cpu_max_num_threads": 8, "memory_layout": "vector", "label": "Vector Memory"},
    # {"cpu_max_num_threads": 16, "memory_layout": "vector", "label": "Vector Memory"},
]
n_grid = [10,100,1000,5000,10000,50000,100000,150000,200000,250000,300000,400000,500000]
n_grid = [1000000]
settings = threads_blocks_managment_settings + memory_management_settings
results = {setting["label"]: [] for setting in settings}
results["GPU"] = []
for n in n_grid:
    print(f"Testing with n{n}")
    for setting in threads_blocks_managment_settings:
        print(f"Testing with {setting} threads")
        taichi_time = compute_taichi_threads_blocks_managment(
            n,
            cpu_max_num_threads=setting["cpu_max_num_threads"],
            block_size=setting["block_size"]
        )
        results[setting["label"]].append(taichi_time)
        print("results", results)

    # Тестирование Taichi для всех настроек использования памяти
    for setting in memory_management_settings:
        taichi_time = compute_taichi_memory_management(
            n,
            cpu_max_num_threads=setting["cpu_max_num_threads"],
            memory_layout=setting["memory_layout"]
        )
        results[setting["label"]].append(taichi_time)

    taichi_time = compute_taichi_gpu(n)
    results["GPU"].append(taichi_time)
    print("results", results)
print(results)