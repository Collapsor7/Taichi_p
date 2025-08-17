import numpy as np
import matplotlib.pyplot as plt
import time

import taichi as ti
from pycparser.ply.yacc import resultlimit

from simple_examples.test_parallel import start_time

ti.init(arch=ti.cpu)

# Параметры системы
m = 0.1  # Масса каждого осциллятора
g = 9.8  # Ускорение свободного падения
dt = 0.01  # Шаг времени
nstep = 25  # Количество шагов

# Параметры связей
Cs = 500.0  # Жесткость при контакте со сферой
Bs = 5.0  # Коэффициент демпфирования при контакте со сферой
R = 0.25  # Радиус сферы


def taichi_compute(N, M):
    sphere_center = ti.Vector([1.0, 1.0, 1.0])  # Центр сферы
    l0 = 1 / (N-1)  # Длина нерастяжимой связи

    # Начальные условия
    x = ti.Vector.field(3, dtype=ti.f32)
    v = ti.Vector.field(3, dtype=ti.f32)
    a = ti.Vector.field(3, dtype=ti.f32)
    ti.root.pointer(ti.i, N).dense(ti.j, M).place(x)
    ti.root.pointer(ti.i, N).dense(ti.j, M).place(v)
    ti.root.pointer(ti.i, N).dense(ti.j, M).place(a)
    forces = ti.Vector.field(3, dtype=ti.f32)
    ti.root.pointer(ti.i, N).dense(ti.j, M).place(forces)
    x_prev = ti.Vector.field(3, dtype=ti.f32)
    ti.root.pointer(ti.i, N).dense(ti.j, M).place(x_prev)

    # Расположение осцилляторов в начальный момент
    @ti.kernel
    def init_x_v():
        for i in range(N):
            for j in range(M):
                x[i, j] = [i * l0 + 0.0, j * l0 + 0.0, 1.5]  # Начальная высота 1.5
                v[i, j] = [0, 0, 0]
                v[i, j] = [0, 0, 0]
                # Предыдущие положения для метода Верле
                x_prev[i, j] = x[i, j] - v[i, j] * dt + 0.5 * a[i, j] * dt ** 2

    init_x_v()

    # Функция для расчета сил
    @ti.kernel
    def compute_forces():
        # Гравитация
        for i in range(N):
            for j in range(M):
                forces[i, j] = [0, 0, - m * g]
        # Взаимодействие со сферой
        for i in range(N):
            for j in range(M):
                forces[i, j] = [0, 0, - m * g]
                r = x[i, j] - sphere_center
                # dist = np.linalg.norm(r)
                dist = ti.sqrt(r.x ** 2 + r.y ** 2 + r.z ** 2)
                if dist < R:  # Если масса внутри сферы
                    normal = r / dist  # Нормаль к поверхности сферы
                    penetration = R - dist  # Глубина проникновения
                    # relative_velocity = np.dot(v[i, j], normal)  # Проекция скорости на нормаль
                    relative_velocity = v[i, j] * normal
                    # Добавляем силу отталкивания и демпфирования
                    forces[i, j] += Cs * penetration * normal + Bs * relative_velocity * normal
                    # Корректируем положение массы на поверхности сферы
                    for _ in range(5):  # Итеративная коррекция
                        r = x[i, j] - sphere_center
                        # dist = np.linalg.norm(r)
                        dist = ti.sqrt(r.x ** 2 + r.y ** 2 + r.z ** 2)
                        if dist < R:
                            normal = r / dist
                            penetration = R - dist
                            x[i, j] += penetration * normal
                    # Разделяем скорость на нормальную и тангенциальную составляющие
                    normal = r / dist
                    # v_normal = np.dot(v[i, j], normal) * normal
                    v_normal = v[i, j].dot(normal) * normal
                    v_tangent = v[i, j] - v_normal
                    v[i, j] = v_tangent  # Сохраняем только тангенциальную составляющую

    # Проекционный метод для восстановления нерастяжимых связей
    @ti.kernel
    def enforce_constraints():
        tolerance = 1e-6  # Допустимая погрешность
        max_iterations = 100  # Максимальное число итераций
        for _ in range(max_iterations):
            max_error = 0.0
            for i in range(N):
                for j in range(M):
                    forces[i, j] = [0, 0, - m * g]
                    # Коррекция связей с соседями
                    if i < N - 1:  # Горизонтальная связь
                        dx = x[i + 1, j] - x[i, j]
                        dist = ti.sqrt(dx.x ** 2 + dx.y ** 2 + dx.z ** 2)
                        error = abs(dist - l0)
                        max_error = max(max_error, error)
                        correction = (l0 - dist) / (2 * dist) * dx
                        x[i + 1, j] += correction
                        x[i, j] -= correction
                    if j < M - 1:  # Вертикальная связь
                        dy = x[i, j + 1] - x[i, j]
                        # dist = np.linalg.norm(dy)
                        dist = ti.sqrt(dy.x ** 2 + dy.y ** 2 + dy.z ** 2)
                        error = abs(dist - l0)
                        max_error = max(max_error, error)
                        correction = (l0 - dist) / (2 * dist) * dy
                        x[i, j + 1] += correction
                        x[i, j] -= correction

    @ti.kernel
    def update_x_v():
        for i in range(N):
            for j in range(M):
                forces[i, j] = [0, 0, - m * g]
                a[i, j] = forces[i, j] / m  # Вычисляем ускорения
                # Метод Верле
                x_next = 2 * x[i, j] - x_prev[i, j] + a[i, j] * dt ** 2
                v_new = (x_next - x_prev[i, j]) / (2 * dt)
                x_prev[i, j] = x[i, j]
                x[i, j] = x_next
                v[i, j] = v_new

    start_time = time.time()

    for step in range(nstep):
        # print(f"STEP: {step}")
        compute_forces()  # Рассчитываем силы
        update_x_v()
        # Применяем проекционный метод
        enforce_constraints()

    elapsed_time = time.time() - start_time
    print(f"Taichi : {N}*{M} calculated finished! elapsed_time:", elapsed_time)

    return elapsed_time

def numpy_compute(N,M):
    sphere_center = np.array([1.0, 1.0, 1.0])  # Центр сферы
    l0 = 1 / (N - 1)

    # Начальные условия
    x = np.zeros((N, M, 3), float)  # Положения (x, y, z)
    v = np.zeros((N, M, 3), float)  # Скорости (vx, vy, vz)
    a = np.zeros((N, M, 3), float)  # Ускорения (ax, ay, az)

    # Расположение осцилляторов в начальный момент
    for i in range(N):
        for j in range(M):
            x[i, j] = [i * l0 + 0.5, j * l0 + 0.5, 1.5]  # Начальная высота 1.5

    # Предыдущие положения для метода Верле
    x_prev = x - v * dt + 0.5 * a * dt ** 2

    # Функция для расчета сил
    def compute_forces(x):
        forces = np.zeros((N, M, 3), float)  # Массив сил для всех масс
        # Гравитация
        for i in range(N):
            for j in range(M):
                forces[i, j, 2] -= m * g
        # Взаимодействие со сферой
        for i in range(N):
            for j in range(M):
                # if fixed[i, j]:  # Пропускаем фиксированные массы
                #     continue
                r = x[i, j] - sphere_center
                dist = np.linalg.norm(r)
                if dist < R:  # Если масса внутри сферы
                    normal = r / dist  # Нормаль к поверхности сферы
                    penetration = R - dist  # Глубина проникновения
                    relative_velocity = np.dot(v[i, j], normal)  # Проекция скорости на нормаль
                    # Добавляем силу отталкивания и демпфирования
                    forces[i, j] += Cs * penetration * normal + Bs * relative_velocity * normal
                    # Корректируем положение массы на поверхности сферы
                    for _ in range(5):  # Итеративная коррекция
                        r = x[i, j] - sphere_center
                        dist = np.linalg.norm(r)
                        if dist < R:
                            normal = r / dist
                            penetration = R - dist
                            x[i, j] += penetration * normal
                    # Разделяем скорость на нормальную и тангенциальную составляющие
                    normal = r / dist
                    v_normal = np.dot(v[i, j], normal) * normal
                    v_tangent = v[i, j] - v_normal
                    v[i, j] = v_tangent  # Сохраняем только тангенциальную составляющую
                    # Проверка, что масса на поверхности
                    assert np.isclose(np.linalg.norm(x[i, j] - sphere_center), R), \
                        f"Ошибка: масса не на поверхности сферы, dist={np.linalg.norm(x[i, j] - sphere_center)}"
        return forces

    # Проекционный метод для восстановления нерастяжимых связей
    def enforce_constraints(x):
        tolerance = 1e-6  # Допустимая погрешность
        max_iterations = 100  # Максимальное число итераций
        for _ in range(max_iterations):
            max_error = 0.0
            for i in range(N):
                for j in range(M):
                    # Коррекция связей с соседями
                    if i < N - 1:  # Горизонтальная связь
                        dx = x[i + 1, j] - x[i, j]
                        dist = np.linalg.norm(dx)
                        error = abs(dist - l0)
                        max_error = max(max_error, error)
                        correction = (l0 - dist) / (2 * dist) * dx
                        x[i + 1, j] += correction
                        x[i, j] -= correction
                    if j < M - 1:  # Вертикальная связь
                        dy = x[i, j + 1] - x[i, j]
                        dist = np.linalg.norm(dy)
                        error = abs(dist - l0)
                        max_error = max(max_error, error)
                        correction = (l0 - dist) / (2 * dist) * dy
                        x[i, j + 1] += correction
                        x[i, j] -= correction
            if max_error < tolerance:
                break

    def update_x_v(forces):
        for i in range(N):
            for j in range(M):
                forces[i, j] = [0, 0, - m * g]
                a[i, j] = forces[i, j] / m  # Вычисляем ускорения
                # Метод Верле
                x_next = 2 * x[i, j] - x_prev[i, j] + a[i, j] * dt ** 2
                v_new = (x_next - x_prev[i, j]) / (2 * dt)
                x_prev[i, j] = x[i, j]
                x[i, j] = x_next
                v[i, j] = v_new

    start_time = time.time()
    for step in range(nstep):
        # print(f"STEP: {step}")
        F = compute_forces(x)  # Рассчитываем силы
        update_x_v(F)
        enforce_constraints(x)

    elapsed_time = time.time() - start_time
    print(f"Numpy : {N}*{M} calculated finished! elapsed_time:", elapsed_time)

    return elapsed_time

size_settings = [
    {"N":5,"M":5},
    {"N":10,"M":10},
    {"N":20,"M":20},
    {"N": 50, "M": 50},
    {"N": 100, "M": 100},
    {"N": 200, "M": 200},
    {"N": 400, "M": 400},
    {"N": 600, "M": 600},
    {"N": 1000, "M": 1000},
]
results={"Taichi":[],"Numpy":[]}

for i in size_settings:
    n = i["N"]
    m = i["M"]
    print(f"Testing with {n}*{m} particles")

    numpy_time = numpy_compute(n,m)
    results["Numpy"].append(numpy_time)
    print(f"{numpy_time = }")

    taichi_time = taichi_compute(n,m)
    results["Taichi"].append(taichi_time)
    print(f"{taichi_time = }")

print(results)

# {'Taichi': [0.08507513999938965, 0.08513903617858887, 0.10660386085510254, 0.26519012451171875, 0.810513973236084, 3.3162009716033936, 12.208231925964355, 26.595811128616333, 75.38270306587219], 'Numpy': [0.010206937789916992, 0.08535909652709961, 0.532865047454834, 3.4858999252319336, 14.071671962738037, 56.39944577217102, 222.83651494979858, 503.4621810913086, 1386.9744591712952]}
