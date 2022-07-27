import taichi as ti
# from FEM_COMMON import FEM
import os
import numpy as np
import scipy.sparse as sp

@ti.data_oriented
class Model:
    def __init__(self, name, Physics, Materials, Omegas=(0, 0, 100), Largedisp=False):
        self.name = name
        self.DE, self.EX, self.PO = Materials['DE'], Materials['EX'], Materials['PO']
        self.OM = np.array(Omegas, dtype=np.float64)

        self.Physics = Physics
        self.Largedisp = Largedisp
        self.Num_Node = self.Physics.Num_Node
        self.Num_Elem = self.Physics.Num_Elem
        self.Num_MaxN = self.Physics.Num_MaxN
        self.Num_Free = self.Num_Node * 3
        self.nnz_K, self.nnz_M = self.Physics.SmNode[-1] * 9, self.Physics.SmNode[-1] * 3

        self.taichi_init()

        print("材料参数---\n\t\t\t密度：{:.2e}, 弹性模量： {:.2e},  泊松比： {:.2e} ".format(self.DE, self.EX, self.PO))
        print("求解规模---\n\t\t\t刚度矩阵内存：{:.3f} Gb,  质量矩阵内存：{:.3f} Gb ".
              format(np.float64(self.nnz_K*1.5)*8/1e9, np.float64(self.nnz_M*1.5)*8/1e9))
        print("\t\t\tCOORD内存：{:.3f} Gb,  KM + MM内存：{:.3f} Gb ".
              format(np.float64(self.Num_Elem*3*8)*8/1e9, np.float64(self.Num_Elem*8*8*10)*8/1e9))


    def taichi_init(self):

        ti.init(arch=ti.cpu, default_fp=ti.f64, default_ip=ti.i32)

        """taichi变量声明"""
        self.GS = ti.field(ti.f64, shape=(2,))  # 存储节点坐标  NCol节点编号
        self.RMat = ti.Matrix.field(3, 1, dtype=ti.f64, shape=8)
        self.DMat = ti.Matrix.field(3, 3, dtype=ti.f64, shape=2)

        self.Node = ti.field(ti.f64, shape=(3, self.Num_Node))  # 存储节点坐标  NCol节点编号
        self.Elem = ti.field(ti.i32, shape=(8, self.Num_Elem))  # 存储单元中节点信息 8*ECol
        self.Nind = ti.field(ti.i32, shape=(self.Num_MaxN,))    # 存储索引
        self.ReNode = ti.field(ti.i32, shape=(self.Physics.SmNode[-1]))
        self.CoNode = ti.field(ti.i32, shape=(self.Num_Node,))      # 与第i个节点在同一单元中的节点个数
        self.SmNode = ti.field(ti.i32, shape=(self.Num_Node + 1,))  # 与第i个节点在同一单元中的节点个数累加

        self.ValK = ti.field(ti.f64, shape=(self.nnz_K,))
        self.ColK = ti.field(ti.i32, shape=(self.nnz_K,))
        self.RowK = ti.field(ti.i32, shape=(self.Num_Free + 1))


        self.ValM = ti.field(ti.f64, shape=(self.nnz_M,))
        self.ColM = ti.field(ti.i32, shape=(self.nnz_M,))
        self.RowM = ti.field(ti.i32, shape=(self.Num_Free + 1))

        self.ValF = ti.field(ti.f64, shape=(3, self.Num_Node))

        """可能需要被优化， 内存占用量太大，是刚度矩阵的1.5倍"""
        self.Coord = ti.Matrix.field(3, 8, dtype=ti.f64, shape=self.Num_Elem)
        self.KMat = ti.Matrix.field(3, 3, dtype=ti.f64, shape=(8, 8, self.Num_Elem))
        self.MMat = ti.field(dtype=ti.f64, shape=(8, 8, self.Num_Elem))

        # 边界条件初始化
        bounds_temp = []
        for bound in self.Physics.Bounds:
            p = bound[0][np.newaxis, :]
            c = np.repeat(np.array(bound[1], dtype=np.float64)[:, np.newaxis], p.shape[-1], axis=-1)
            bounds_temp.append(np.concatenate((p, c), axis=0))
        if len(bounds_temp):
            bounds_temp = np.concatenate(bounds_temp, axis=-1)
        else:
            print("不存在边界条件文件，将自动添加边界条件！")
            bounds_temp = np.zeros((4, 1), dtype=np.float64)

        forces_temp = []
        for force in self.Physics.Forces:
            p = force[0][np.newaxis, :]
            v = np.repeat(np.array(force[1], dtype=np.int)[:, np.newaxis], p.shape[-1], axis=-1)
            forces_temp.append(np.concatenate((p, v), axis=0))
        if len(forces_temp):
            forces_temp = np.concatenate(forces_temp, axis=-1)
        else:
            print("不存在载荷信息文件，将自动添加载荷信息！")
            forces_temp = np.zeros((4, 1), dtype=np.float64)


        self.Forces = ti.field(ti.f64, shape=(4, forces_temp.shape[-1]))
        self.Bounds = ti.field(ti.f64, shape=(4, bounds_temp.shape[-1]))
        self.OMEGAs = ti.field(ti.f64, shape=(3,)) # 全局旋转角速度


        """taichi变量赋值"""
        self.GS[0], self.GS[1] = -0.57735026918962, 0.57735026918962
        # FEM网格赋值
        self.Node.from_numpy(self.Physics.Node[-3:, :])
        self.Elem.from_numpy(self.Physics.Elem[-8:, :])
        self.Nind.from_numpy(self.Physics.Nind)
        self.ReNode.from_numpy(self.Physics.ReNode)
        self.CoNode.from_numpy(self.Physics.CoNode)
        self.SmNode.from_numpy(self.Physics.SmNode)
        # FEM边界条件和载荷赋值
        self.OMEGAs.from_numpy(self.OM)
        self.Bounds.from_numpy(bounds_temp)
        self.Forces.from_numpy(forces_temp)
        # FEM计算矩阵赋值
        self.FEM_CONST()


        """taichi预编译"""
        self.InstallIndexK(0)
        self.InstallIndexM(0)
        self.InstallMatrixK(0)
        self.InstallMatrixM(0)
        self.InstallVectorF(0)
        self.ModifyK(0)
        self.LoadF(0)

    def Get_MatrixKL(self):

        self.InstallIndexK(self.Num_Node)
        self.InstallMatrixK(self.Num_Elem)
        self.ModifyK(self.Bounds.shape[-1])
        ValK = self.ValK.to_numpy()
        RowK = self.RowK.to_numpy()
        ColK = self.ColK.to_numpy()
        K = sp.csr_matrix((ValK, ColK, RowK), shape=(self.Num_Free, self.Num_Free), dtype=np.float64)
        return K

    def Get_MatrixM(self):

        self.InstallIndexM(self.Num_Node)
        self.InstallMatrixM(self.Num_Elem)
        ValM = self.ValM.to_numpy()
        RowM = self.RowM.to_numpy()
        ColM = self.ColM.to_numpy()
        M = sp.csr_matrix((ValM, ColM, RowM), shape=(self.Num_Free, self.Num_Free), dtype=np.float64)
        return M



    def Get_VectorF(self):

        self.InstallVectorF(self.Num_Elem)
        self.LoadF(self.Forces.shape[-1])

        return self.ValF.to_numpy().T.flatten()



    @ti.kernel
    def InstallMatrixK(self, Num_Elem: ti.i32):  # 组装稀疏刚度矩阵
        for i in range(Num_Elem):
            self.UnitMatrixK(i)
            for j in range(8):
                IndexRow = self.Nind[self.Elem[j, i]]
                for k in range(8):
                    IndexCol = self.Nind[self.Elem[k, i]]
                    for m in range(self.CoNode[IndexRow]):
                        IndexRe = self.SmNode[IndexRow] + m
                        if IndexCol == self.ReNode[IndexRe]:
                            for p, q in ti.static(ti.ndrange(3, 3)):
                                IndexK = 9 * self.SmNode[IndexRow] + 3 * p * self.CoNode[IndexRow] + m * 3 + q
                                self.ColK[IndexK] = IndexCol * 3 + q
                                self.ValK[IndexK] += self.KMat[j, k, i][p, q]
                            break

    @ti.kernel
    def InstallIndexK(self, Num_Node: ti.i32):

        for i in range(Num_Node):
            self.RowK[3*i+1] = self.CoNode[i] * 3 + self.SmNode[i] * 9
            self.RowK[3*i+2] = self.CoNode[i] * 3 * 2 + self.SmNode[i] * 9
            self.RowK[3*i+3] = self.CoNode[i] * 3 * 3 + self.SmNode[i] * 9



    @ti.kernel
    def InstallMatrixM(self, Num_Elem: ti.i32):  # 组装质量矩阵（稀疏）
        for i in range(Num_Elem):
            self.UnitMatrixM(i)
            for j in range(8):
                IndexRow = self.Nind[self.Elem[j, i]]
                for k in range(8):
                    IndexCol = self.Nind[self.Elem[k, i]]
                    for m in range(self.CoNode[IndexRow]):
                        IndexRe = self.SmNode[IndexRow] + m
                        if IndexCol == self.ReNode[IndexRe]:
                            for p in ti.static(range(3)):
                                IndexM = 3 * self.SmNode[IndexRow] + p * self.CoNode[IndexRow] + m
                                self.ValM[IndexM] += self.MMat[j, k, i]
                                self.ColM[IndexM] = IndexCol * 3 + p
                            break


    @ti.kernel
    def InstallIndexM(self, Num_Node: ti.i32):

        for i in range(Num_Node):
            self.RowM[3*i+1] = self.CoNode[i] + self.SmNode[i] * 3
            self.RowM[3*i+2] = self.CoNode[i] * 2 + self.SmNode[i] * 3
            self.RowM[3*i+3] = self.CoNode[i] * 3 + self.SmNode[i] * 3


    @ti.kernel
    def LoadF(self, Num_Forces: ti.i32):
        for i in range(Num_Forces):
            point = int(self.Forces[0, i])
            self.ValF[0, self.Nind[point]] += self.Forces[1, i]
            self.ValF[1, self.Nind[point]] += self.Forces[2, i]
            self.ValF[2, self.Nind[point]] += self.Forces[3, i]


    @ti.kernel
    def InstallVectorF(self, Num_Elem: ti.i32):
        for i in range(Num_Elem):
            for g in range(8):
                rtr = self.RMat[g]
                Nr = self.shape_func(rtr)
                DNr = self.shape_derv(rtr)
                R = self.Coord[i] @ Nr
                J = self.Coord[i] @ DNr
                F = self.UnitVectorF(R)
                J_det = self.DE * J.determinant()
                for k in range(8):
                    N = self.installLocalN(k, Nr)
                    indexF = self.Nind[self.Elem[k, i]]
                    self.ValF[0, indexF] += J_det * N * F[0]
                    self.ValF[1, indexF] += J_det * N * F[1]
                    self.ValF[2, indexF] += J_det * N * F[2]


    @ti.kernel
    def ModifyK(self, Num_Bounds: ti.i32):
            for i in range(Num_Bounds):
                point = int(self.Bounds[0, i])
                Index = self.Nind[point]
                for k in range(self.CoNode[Index]):
                    if self.ReNode[self.SmNode[Index]+k] == Index:
                        for j in range(3):
                            if self.Bounds[j+1, i] != -1:
                                self.ValK[9*self.SmNode[Index]+3*j*self.CoNode[Index]+k*3+j] = 1e20
                        break


    @ti.func
    def UnitMatrixK(self, i):  # 建立单元刚度矩阵
        for g in range(8):
            rtr = self.RMat[g]  # 局部坐标
            DNr = self.shape_derv(rtr)
            J = self.Coord[i] @ DNr
            J_ = J.inverse()
            DN = DNr @ J_
            J_det = J.determinant()
            for j, k in ti.ndrange(8, 8):
                B11, B12 = self.installLocalB(j, DN)
                B21, B22 = self.installLocalB(k, DN)
                self.KMat[j, k, i] += J_det * (B11.transpose() @ self.DMat[0] @ B21 + B12.transpose() @ self.DMat[1] @ B22)

    @ti.func
    def UnitMatrixM(self, i):  # 各单元质量矩阵
        for g in range(8):
            rtr = self.RMat[g]
            Nr = self.shape_func(rtr)
            DNr = self.shape_derv(rtr)
            J = self.Coord[i] @ DNr
            J_det = self.DE * J.determinant()
            for j, k in ti.ndrange(8, 8):
                N1 = self.installLocalN(j, Nr)
                N2 = self.installLocalN(k, Nr)
                self.MMat[j, k, i] += J_det * N1 * N2



    @ti.func
    def UnitVectorF(self, R):
        F = ti.Matrix([[0.0], [0.0], [0.0]])
        F[0] = self.OMEGAs[1] * self.OMEGAs[1] * R[0] + self.OMEGAs[2] * self.OMEGAs[2] * R[0] - \
               self.OMEGAs[0] * self.OMEGAs[1] * R[1] - self.OMEGAs[0] * self.OMEGAs[2] * R[2]
        F[1] = self.OMEGAs[0] * self.OMEGAs[0] * R[1] + self.OMEGAs[2] * self.OMEGAs[2] * R[1] - \
               self.OMEGAs[1] * self.OMEGAs[2] * R[2] - self.OMEGAs[1] * self.OMEGAs[0] * R[0]
        F[2] = self.OMEGAs[1] * self.OMEGAs[1] * R[2] + self.OMEGAs[0] * self.OMEGAs[0] * R[2] - \
               self.OMEGAs[2] * self.OMEGAs[0] * R[0] - self.OMEGAs[2] * self.OMEGAs[1] * R[1]
        return F


    @ti.func
    def installLocalB(self, index, DN):
        B1, B2 = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), \
                 ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        if index == 0:
            B1[0, 0], B1[1, 1], B1[2, 2] = DN[0, 0], DN[0, 1], DN[0, 2]
            B2[0, 0], B2[0, 1] = DN[0, 1], DN[0, 0]
            B2[1, 1], B2[1, 2] = DN[0, 2], DN[0, 1]
            B2[2, 0], B2[2, 2] = DN[0, 2], DN[0, 0]
        elif index == 1:
            B1[0, 0], B1[1, 1], B1[2, 2] = DN[1, 0], DN[1, 1], DN[1, 2]
            B2[0, 0], B2[0, 1] = DN[1, 1], DN[1, 0]
            B2[1, 1], B2[1, 2] = DN[1, 2], DN[1, 1]
            B2[2, 0], B2[2, 2] = DN[1, 2], DN[1, 0]
        elif index == 2:
            B1[0, 0], B1[1, 1], B1[2, 2] = DN[2, 0], DN[2, 1], DN[2, 2]
            B2[0, 0], B2[0, 1] = DN[2, 1], DN[2, 0]
            B2[1, 1], B2[1, 2] = DN[2, 2], DN[2, 1]
            B2[2, 0], B2[2, 2] = DN[2, 2], DN[2, 0]
        elif index == 3:
            B1[0, 0], B1[1, 1], B1[2, 2] = DN[3, 0], DN[3, 1], DN[3, 2]
            B2[0, 0], B2[0, 1] = DN[3, 1], DN[3, 0]
            B2[1, 1], B2[1, 2] = DN[3, 2], DN[3, 1]
            B2[2, 0], B2[2, 2] = DN[3, 2], DN[3, 0]
        elif index == 4:
            B1[0, 0], B1[1, 1], B1[2, 2] = DN[4, 0], DN[4, 1], DN[4, 2]
            B2[0, 0], B2[0, 1] = DN[4, 1], DN[4, 0]
            B2[1, 1], B2[1, 2] = DN[4, 2], DN[4, 1]
            B2[2, 0], B2[2, 2] = DN[4, 2], DN[4, 0]
        elif index == 5:
            B1[0, 0], B1[1, 1], B1[2, 2] = DN[5, 0], DN[5, 1], DN[5, 2]
            B2[0, 0], B2[0, 1] = DN[5, 1], DN[5, 0]
            B2[1, 1], B2[1, 2] = DN[5, 2], DN[5, 1]
            B2[2, 0], B2[2, 2] = DN[5, 2], DN[5, 0]
        elif index == 6:
            B1[0, 0], B1[1, 1], B1[2, 2] = DN[6, 0], DN[6, 1], DN[6, 2]
            B2[0, 0], B2[0, 1] = DN[6, 1], DN[6, 0]
            B2[1, 1], B2[1, 2] = DN[6, 2], DN[6, 1]
            B2[2, 0], B2[2, 2] = DN[6, 2], DN[6, 0]
        elif index == 7:
            B1[0, 0], B1[1, 1], B1[2, 2] = DN[7, 0], DN[7, 1], DN[7, 2]
            B2[0, 0], B2[0, 1] = DN[7, 1], DN[7, 0]
            B2[1, 1], B2[1, 2] = DN[7, 2], DN[7, 1]
            B2[2, 0], B2[2, 2] = DN[7, 2], DN[7, 0]
        return B1, B2

    @ti.func
    def installLocalN(self, index, Nr):
        N = 0.0
        if index == 0:
            N = Nr[0]
        elif index == 1:
            N = Nr[1]
        elif index == 2:
            N = Nr[2]
        elif index == 3:
            N = Nr[3]
        elif index == 4:
            N = Nr[4]
        elif index == 5:
            N = Nr[5]
        elif index == 6:
            N = Nr[6]
        elif index == 7:
            N = Nr[7]
        return N


    @ti.func
    def shape_func(self, rtr):
        Nr = ti.Matrix.zero(dt=ti.f64, m=1, n=8)
        Nr[0] = 0.125 * (1 + rtr[0]) * (1 + rtr[1]) * (1 - rtr[2])
        Nr[1] = 0.125 * (1 - rtr[0]) * (1 + rtr[1]) * (1 - rtr[2])
        Nr[2] = 0.125 * (1 - rtr[0]) * (1 - rtr[1]) * (1 - rtr[2])
        Nr[3] = 0.125 * (1 + rtr[0]) * (1 - rtr[1]) * (1 - rtr[2])
        Nr[4] = 0.125 * (1 + rtr[0]) * (1 + rtr[1]) * (1 + rtr[2])
        Nr[5] = 0.125 * (1 - rtr[0]) * (1 + rtr[1]) * (1 + rtr[2])
        Nr[6] = 0.125 * (1 - rtr[0]) * (1 - rtr[1]) * (1 + rtr[2])
        Nr[7] = 0.125 * (1 + rtr[0]) * (1 - rtr[1]) * (1 + rtr[2])
        return Nr

    @ti.func
    def shape_derv(self, rtr):   # 形函数对局部坐标偏导
        DNr = ti.Matrix.zero(dt=ti.f64, m=3, n=8)
        DNr[2, 0] = 0.125 * (-1) * (1 - rtr[1]) * (1 - rtr[2])
        DNr[2, 1] = 0.125 * (1 - rtr[0]) * (-1) * (1 - rtr[2])
        DNr[2, 2] = 0.125 * (1 - rtr[0]) * (1 - rtr[1]) * (-1)

        DNr[6, 0] = 0.125 * (-1) * (1 - rtr[1]) * (1 + rtr[2])
        DNr[6, 1] = 0.125 * (1 - rtr[0]) * (-1) * (1 + rtr[2])
        DNr[6, 2] = 0.125 * (1 - rtr[0]) * (1 - rtr[1]) * (+1)

        DNr[7, 0] = 0.125 * (+1) * (1 - rtr[1]) * (1 + rtr[2])
        DNr[7, 1] = 0.125 * (1 + rtr[0]) * (-1) * (1 + rtr[2])
        DNr[7, 2] = 0.125 * (1 + rtr[0]) * (1 - rtr[1]) * (+1)

        DNr[3, 0] = 0.125 * (+1) * (1 - rtr[1]) * (1 - rtr[2])
        DNr[3, 1] = 0.125 * (1 + rtr[0]) * (-1) * (1 - rtr[2])
        DNr[3, 2] = 0.125 * (1 + rtr[0]) * (1 - rtr[1]) * (-1)

        DNr[1, 0] = 0.125 * (-1) * (1 + rtr[1]) * (1 - rtr[2])
        DNr[1, 1] = 0.125 * (1 - rtr[0]) * (+1) * (1 - rtr[2])
        DNr[1, 2] = 0.125 * (1 - rtr[0]) * (1 + rtr[1]) * (-1)

        DNr[5, 0] = 0.125 * (-1) * (1 + rtr[1]) * (1 + rtr[2])
        DNr[5, 1] = 0.125 * (1 - rtr[0]) * (+1) * (1 + rtr[2])
        DNr[5, 2] = 0.125 * (1 - rtr[0]) * (1 + rtr[1]) * (+1)

        DNr[4, 0] = 0.125 * (+1) * (1 + rtr[1]) * (1 + rtr[2])
        DNr[4, 1] = 0.125 * (1 + rtr[0]) * (+1) * (1 + rtr[2])
        DNr[4, 2] = 0.125 * (1 + rtr[0]) * (1 + rtr[1]) * (+1)

        DNr[0, 0] = 0.125 * (+1) * (1 + rtr[1]) * (1 - rtr[2])
        DNr[0, 1] = 0.125 * (1 + rtr[0]) * (+1) * (1 - rtr[2])
        DNr[0, 2] = 0.125 * (1 + rtr[0]) * (1 + rtr[1]) * (-1)

        return DNr

    @ti.kernel
    def FEM_CONST(self):

        # 初始化高斯积分点
        self.RMat[0][0], self.RMat[0][1], self.RMat[0][2] = self.GS[0], self.GS[0], self.GS[0]
        self.RMat[1][0], self.RMat[1][1], self.RMat[1][2] = self.GS[0], self.GS[0], self.GS[1]
        self.RMat[2][0], self.RMat[2][1], self.RMat[2][2] = self.GS[0], self.GS[1], self.GS[0]
        self.RMat[3][0], self.RMat[3][1], self.RMat[3][2] = self.GS[0], self.GS[1], self.GS[1]
        self.RMat[4][0], self.RMat[4][1], self.RMat[4][2] = self.GS[1], self.GS[0], self.GS[0]
        self.RMat[5][0], self.RMat[5][1], self.RMat[5][2] = self.GS[1], self.GS[0], self.GS[1]
        self.RMat[6][0], self.RMat[6][1], self.RMat[6][2] = self.GS[1], self.GS[1], self.GS[0]
        self.RMat[7][0], self.RMat[7][1], self.RMat[7][2] = self.GS[1], self.GS[1], self.GS[1]

        # 初始化应力应变矩阵
        self.DMat[0][0, 0] = self.EX * (1 - self.PO) / (1 + self.PO) / (1 - 2 * self.PO)
        self.DMat[0][1, 1] = self.EX * (1 - self.PO) / (1 + self.PO) / (1 - 2 * self.PO)
        self.DMat[0][2, 2] = self.EX * (1 - self.PO) / (1 + self.PO) / (1 - 2 * self.PO)

        self.DMat[0][0, 1] = self.EX * self.PO / (1 + self.PO) / (1 - 2 * self.PO)
        self.DMat[0][0, 2] = self.EX * self.PO / (1 + self.PO) / (1 - 2 * self.PO)
        self.DMat[0][1, 2] = self.EX * self.PO / (1 + self.PO) / (1 - 2 * self.PO)

        self.DMat[0][1, 0] = self.EX * self.PO / (1 + self.PO) / (1 - 2 * self.PO)
        self.DMat[0][2, 0] = self.EX * self.PO / (1 + self.PO) / (1 - 2 * self.PO)
        self.DMat[0][2, 1] = self.EX * self.PO / (1 + self.PO) / (1 - 2 * self.PO)

        self.DMat[1][0, 0] = self.EX / 2 / (1 + self.PO)
        self.DMat[1][1, 1] = self.EX / 2 / (1 + self.PO)
        self.DMat[1][2, 2] = self.EX / 2 / (1 + self.PO)

        for i in range(self.Num_Elem):
            for j, k in ti.static(ti.ndrange(8, 3)):
                self.Coord[i][k, j] = self.Node[k, self.Nind[self.Elem[j, i]]]

