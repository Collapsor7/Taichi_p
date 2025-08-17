import gmsh
import numpy as np
import os

# 参数定义
A = 1.0   # 沿 X 轴尺寸, m
B = 1.0   # 沿 Y 轴尺寸, m
H = np.array([0.001, 0.001, 0.001])  # 各层厚度, m
N = H.shape[0]  # 层数
Nx = 50  # X 方向网格数（仅用于参考）
Ny = 50  # Y 方向网格数（仅用于参考）
Nz = 1


gmsh.initialize()
gmsh.model.add("Model_1_gmsh_occ")

# 创建底面矩形
gmsh.model.occ.addRectangle(x=0, y=0, z=0, dx=A, dy=B, tag=1)
gmsh.model.occ.synchronize()

# 第一层挤出
Layer_mesh = gmsh.model.occ.extrude(dimTags=[(2, 1)],
                                    dx=0, dy=0, dz=H[0],
                                    numElements=[1],
                                    heights=[1],
                                    recombine=True)
gmsh.model.occ.synchronize()


for i in range(1, N):
    Layer_mesh = gmsh.model.occ.extrude(dimTags=[Layer_mesh[0]],
                                        dx=0, dy=0, dz=H[i],
                                        numElements=[1],
                                        heights=[1],
                                        recombine=True)
gmsh.model.occ.synchronize()
gmsh.fltk.run()


Points = gmsh.model.occ.getEntities(dim=0)
for point in Points:
    x, y, z = gmsh.model.occ.getCenterOfMass(point[0], point[1])
    size = 0.001 + 0.002 * abs(x - A/2)  # 例如在 x=A/2 附近更细
    gmsh.model.mesh.setSize([point], size)

gmsh.model.mesh.setTransfiniteAutomatic(recombine=True)
gmsh.option.setNumber("Mesh.Algorithm", 1)
gmsh.option.setNumber("Mesh.Algorithm3D", 1)
gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)

# 生成三维网格
gmsh.model.mesh.generate(3)

gmsh.fltk.run()

gmsh.finalize()