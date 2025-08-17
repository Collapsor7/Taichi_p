# 重新导入库并绘图
import matplotlib.pyplot as plt
import numpy as np

# 数据
taichi_times = [0.08507513999938965, 0.08513903617858887, 0.10660386085510254, 0.26519012451171875, 0.810513973236084, 3.3162009716033936, 12.208231925964355, 26.595811128616333, 75.38270306587219]
numpy_times = [0.010206937789916992, 0.08535909652709961, 0.532865047454834, 3.4858999252319336, 14.071671962738037, 56.39944577217102, 222.83651494979858, 503.4621810913086, 1386.9744591712952]
labels = ["5×5", "10×10", "20×20", "50×50", "100×100", "200×200", "400×400", "600×600", "1000×1000"]
speedup = [t / n for t, n in zip(numpy_times, taichi_times)]

# 图形设置，总宽高比 2:1，每张子图为1:1
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # 宽12，高6，即比例为2:1

# 左图：运行时间
ax1.plot(labels, taichi_times, label='Taichi(CPU)', marker='o')
ax1.plot(labels, numpy_times, label='NumPy', marker='o')
ax1.set_yscale("log")
ax1.set_xlabel("Размер сетки")
ax1.set_ylabel("Время выполнения (сек)")
ax1.set_title("Сравнение времени выполнения")
ax1.legend()
ax1.grid(True)

# 右图：加速比
ax2.plot(labels, speedup, marker='o', color='green')
ax2.set_xlabel("Размер сетки")
ax2.set_ylabel("Ускорение (NumPy / Taichi)")
ax2.set_title("Коэффициент ускорения")
ax2.grid(True)

# ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)

# ax1.set_xticks(x)
ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)

plt.tight_layout()
plt.show()