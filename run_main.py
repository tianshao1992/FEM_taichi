import time
import numpy as np
import taichi as ti
import sys

from COMMON import Logger
from READ_physics import Read
from FEM_model_csr import Model
from SOLVE_static import Static_solver
from SOLVE_eigen import Eigen_solver
from POST_process import Post
import os


if __name__ == "__main__":

    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    name = 'long'
    sys.stdout = Logger("Results\\" + name + "\\log.txt")  # 这里将Log输出到指定文件中
    Materials = {'DE': 7.8e-9, 'EX': 200000.0, 'PO': 0.3}  # 材料物性
    Omegas = (0, 0, 100)  # 全局角速度

    """ 网格/边界条件读入 """
    PHYS = Read(name=name)  # 网格文件名，name 后缀为.cdb

    """ 有限元刚度/质量矩阵 """
    FEMS = Model(name, PHYS, Materials, Omegas=Omegas, Largedisp=False)


    """" 求解静力学问题 """
    solver = Static_solver(FEMS, 0)
    Displaces = solver.Solve_cpu()

    """" 求解模态问题 """
    solver = Eigen_solver(FEMS, 0)
    Frequency, Vibrations = solver.Solve_cpu(num_eigen=100)

    ti.reset()

    """ 后处理 """
    Displaces = Displaces.reshape((PHYS.Num_Node, 3, -1))   # 场的维度包括三个：节点数，变量个数，载荷步或模态数
    POST = Post(name, PHYS, Materials, Displaces)
    Stress = POST.Get_Stress()[:, :, np.newaxis]

    Fields = np.concatenate((Displaces, Stress), axis=1)
    Fields_name = ['u', 'v', 'w', 's-xx', 's-yy', 's-zz', 's-xy', 's-yz', 's-xz', '1-prin', '2-prin', '3-prin', 'von-mises']
    POST.Output_field_BIN('statics', Fields, Fields_name)


    Fields = Vibrations.reshape((PHYS.Num_Node, 3, -1))
    Fields_name = ['u', 'v', 'w']
    POST.Output_field_BIN('vibration', Fields, Fields_name)
