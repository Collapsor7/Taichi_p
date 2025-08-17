import taichi as ti
import numpy as np
import time
import matplotlib.pyplot as plt
from IPython.terminal.ipapp import flags

#
# # Инициализация Taichi
ti.init(arch=ti.cpu)
#
# # Размеры массивов для тестирования
particle_counts = [10, 100, 500, 10 ** 3, 2 * 10 ** 3, 5 * 10 ** 3, 10 ** 4, 2 * 10 ** 4]

# Настройки Taichi для тестирования
threads_blocks_managment_settings = [
    {"cpu_max_num_threads": 1, "block_size": 1, "label": "Threads=1, Block=1"},
    {"cpu_max_num_threads": 8, "block_size": 16, "label": "Threads=8, Block=16"},
    {"cpu_max_num_threads": 8, "block_size": 32, "label": "Threads=8, Block=32"},
    {"cpu_max_num_threads": 16, "block_size": 64, "label": "Threads=16, Block=64"},
]

memory_management_settings = [
    {"cpu_max_num_threads": 8, "memory_layout": "dense", "label": "Dense Memory"},
    {"cpu_max_num_threads": 8, "memory_layout": "pointer", "label": "Pointer Memory"},
    {"cpu_max_num_threads": 8, "memory_layout": "vector", "label": "Vector Memory"},
]
#
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
#
#
# # Функция для вычисления сил в Taichi
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

#
# # Сбор данных для графиков
settings = threads_blocks_managment_settings + memory_management_settings
results = {setting["label"]: [] for setting in settings}
results["NumPy"] = []

print(f"{results = }")

for n in particle_counts:
    print(f"Testing with {n} particles")

    # Тестирование NumPy
    numpy_time = compute_numpy(n)
    results["NumPy"].append(numpy_time)

    print(f"{numpy_time = }")
    # Тестирование Taichi для всех настроек многопоточности
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

print(results)

# Построение графиков
plt.figure(figsize=(12, 6))

# График времени выполнения
plt.subplot(1, 2, 1)
for label, times in results.items():
    plt.plot(particle_counts, times, label=label, marker='o')
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Число частиц в массиве")
plt.ylabel("Время выполнения (сек)")
plt.title("Зависимость времени выполнения от числа частиц в массиве")
plt.legend()

# График ускорения относительно NumPy
plt.subplot(1, 2, 2)
for label, times in results.items():
    if label != "NumPy":
        speedup = [numpy_t / t for numpy_t, t in zip(results["NumPy"], times)]
        plt.plot(particle_counts, speedup, label=label, marker='o')
plt.xscale("log")
plt.xlabel("Число частиц в массиве")
plt.ylabel("Ускорение относительно NumPy)")
plt.title("Зависимость ускорения от числа частиц в массиве")
plt.legend()

plt.tight_layout()
plt.show()

# run_experiments.py
# import subprocess
#
# results = {}
#
# settings = threads_blocks_managment_settings + memory_management_settings
# results = {setting["label"]: [] for setting in settings}
# results["NumPy"] = []
#
# print(f"{results = }")
#
# for n in particle_counts:
#     print(f"Testing with {n} particles")
#
#     # Тестирование NumPy
#     # numpy_time = compute_numpy(n)
#     # results["NumPy"].append(numpy_time)
#
#
#     # print(f"{numpy_time = }")
#     # Тестирование Taichi для всех настроек многопоточности
#     for setting in threads_blocks_managment_settings:
#         print(f"Testing with {setting} threads")
#         cmd = [
#             "python", "simulate.py",
#             str(setting["cpu_max_num_threads"]),
#             str(setting["block_size"]),
#             str(n),
#             str(nstep)
#         ]
#         proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#         out, err = proc.communicate()
#         print("out",out,"err",err)
#         # print("out[2]",out[2],type(out[2]))
#         if err:
#             print("Error in", setting["label"], ":\n", err)
#         result_line = None
#         for line in out.splitlines():
#             if line.startswith("Elapsed time:"):
#                 result_line = line[len("Elapsed time:"):]  # 去掉标记
#                 break
#         # print(result_line)
#         out_result = float(result_line)
#         # print(out_result,type(out_result))
#         # results[setting["label"]] = out_result
#         print("Output for", setting["label"], ":", out_result)
#         taichi_time = out_result
#         results[setting["label"]].append(taichi_time)
#         print("results", results)
#
#     # Тестирование Taichi для всех настроек использования памяти
#     for setting in memory_management_settings:
#         cmd = [
#             "python", "simulate.py",
#             str(setting["cpu_max_num_threads"]),
#             str(setting["block_size"]),
#             str(n),
#             str(nstep)
#         ]
#         proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#         out, err = proc.communicate()
#         if err:
#             print("Error in", setting["label"], ":\n", err)
#         results[setting["label"]] = out
#         print("Output for", setting["label"], ":\n", out)
#         taichi_time = out
#         if err:
#             print("Error in", setting["label"], ":\n", err)
#         results[setting["label"]].append(taichi_time)
#
# # for cfg in configs:
# #     print("Running configuration:", cfg["label"])
# #     # 构造命令
# #     cmd = [
# #         "python", "simulate.py",
# #         str(cfg["cpu_max_num_threads"]),
# #         str(cfg["block_size"]),
# #         str(n),
# #         str(nstep)
# #     ]
# #     # 运行子进程
# #     proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
# #     out, err = proc.communicate()
# #     if err:
# #         print("Error in", cfg["label"], ":\n", err)
# #     results[cfg["label"]] = out
# #     print("Output for", cfg["label"], ":\n", out)
#
# plt.figure(figsize=(12, 6))
#
# # График времени выполнения
# plt.subplot(1, 2, 1)
# for label, times in results.items():
#     plt.plot(particle_counts, times, label=label, marker='o')
# plt.xscale("log")
# plt.yscale("log")
# plt.xlabel("Число частиц в массиве")
# plt.ylabel("Время выполнения (сек)")
# plt.title("Зависимость времени выполнения от числа частиц в массиве")
# plt.legend()
#
# # График ускорения относительно NumPy
# plt.subplot(1, 2, 2)
# for label, times in results.items():
#     if label != "NumPy":
#         speedup = [numpy_t / t for numpy_t, t in zip(results["NumPy"], times)]
#         plt.plot(particle_counts, speedup, label=label, marker='o')
# plt.xscale("log")
# plt.xlabel("Число частиц в массиве")
# plt.ylabel("Ускорение относительно NumPy)")
# plt.title("Зависимость ускорения от числа частиц в массиве")
# plt.legend()
#
# plt.tight_layout()
# plt.show()