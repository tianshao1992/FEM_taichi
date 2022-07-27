import numpy as np
import time
# import pymatsolver
from pyMKL import pardisoSolver

class Static_solver:
    def __init__(self, Model, Qtype=0):
        self.Qtype = Qtype   # 取1时求解大变形非线性静力学问题
        self.Model = Model


    def Solve_cpu(self, max_errs=1e-4, max_iters=20):

        if self.Qtype == 0:

            print('载荷向量添加......')
            t1 = time.time()
            F = self.Model.Get_VectorF()
            print('\t\t\t耗时 {:.2f} 秒'.format(time.time() - t1))

            print("开始求解线性静力学问题.....")

            if not hasattr(self.Model, 'KL'):
                t1 = time.time()
                self.Model.KL = self.Model.Get_MatrixKL()
                print('\t\t\t组装耗时 {:.2f} 秒'.format(time.time() - t1))

            t1 = time.time()
            # self.Model.K_inv = pymatsolver.Pardiso(self.Model.KL, is_symmetric=True)
            # U = self.Model.K_inv * F
            self.Model.linear_solver = pardisoSolver(self.Model.KL, mtype=2)
            self.Model.linear_solver.factor()
            U = self.Model.linear_solver.solve(F)
            print('\t\t\t求解耗时 {:.2f} 秒'.format(time.time() - t1))

            RF = self.Model.KL * U - F
            error = np.linalg.norm(RF) / np.linalg.norm(F)
            print("\t\t\t相对误差 {:.3e}".format(error))


        elif self.Qtype == 1:
            print('开始求解大变形非线性问题.....')
            U = np.zeros(shape=(3 * self.Model.Num_Node,))

            print('载荷向量添加......')
            t1 = time.time()
            F = self.Model.Get_VectorF()
            print('\t\t\t耗时 {:.2f} 秒'.format(time.time() - t1))

            step = 0
            while True:
                print('迭代次数:', step+1, '-----')
                t1 = time.time()
                self.Model.LDinit(displace=U)
                self.Model.KT, self.Model.KM = self.Model.Get_MatrixKN()
                print('\t\t\t组装耗时 {:.2f} 秒'.format(time.time() - t1))

                RF = self.Model.KM * U - F
                error = np.linalg.norm(RF) / np.linalg.norm(F)
                print("\t\t\t相对误差 {:.3e}".format(error))
                if error <= max_errs or step == max_iters:
                    if step == max_iters:
                        print('牛顿迭代法无法在给定最大迭代次数内收敛！')
                    break

                t1 = time.time()
                # self.Model.K_inv = pymatsolver.Pardiso(self.Model.KT, is_symmetric=True)
                # dU = self.Model.K_inv * RF
                # self.Model.K_inv.clean()
                self.Model.linear_solver = pardisoSolver(self.Model.KT, mtype=2)
                self.Model.linear_solver.factor()
                dU = self.Model.linear_solver.solve(F)
                U = U - dU
                print('\t\t\t求解耗时 {:.2f} 秒'.format(time.time() - t1),'\n')
                self.Model.linear_solver.clear()
                step += 1

        return U