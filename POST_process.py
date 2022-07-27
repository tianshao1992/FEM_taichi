import numpy as np
import pandas as pd
import os
import numpy.linalg as nl
import taichi as ti
import time
from ctypes import *
import numpy.ctypeslib as nc
import os

dll_path = os.getcwd() + "\\utilize\\Dll1\\x64\\Debug\\Dll1.dll"
DL = cdll.LoadLibrary(dll_path)

@ti.data_oriented
class Post(object):

    def __init__(self, name, Physics, Materials, Solutions):

        self.name = name
        self.path = "Results\\" + name + "\\"
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.DE, self.EX, self.PO = Materials['DE'], Materials['EX'], Materials['PO']

        self.Physics = Physics
        self.Num_Node = self.Physics.Node.shape[-1]
        self.Num_Elem = self.Physics.Elem.shape[-1]
        self.Solutions = Solutions.squeeze().T

        self.taichi_init()

    def Output_field_BIN(self, file_name, Fields, Field_names):

        print(file_name + "输出为tecplot二进制文件")
        t1 = time.time()

        Elem = self.Physics.Elem[-8:]
        Node = self.Physics.Node[-3:]
        Nind = self.Physics.Nind
        data_out = np.concatenate((Node.T, Fields.transpose(0, 2, 1).reshape((Node.shape[-1], -1))), axis=-1)
        valnum = data_out.shape[-1]
        data_out = data_out.T.flatten()

        VariableNames = "x y z "
        for i in range(Fields.shape[-1]):
            for name in Field_names:
                VariableNames = VariableNames + name + str(i+1) + " "

        Title = "Solution"
        FileName = self.path + file_name + '.plt'
        ScratchDir = "."

        Title = bytes(Title, "gbk")
        VariableNames = bytes(VariableNames, "gbk")
        FileName = bytes(FileName, "gbk")
        ScratchDir = bytes(ScratchDir, "gbk")

        temp = []
        for i in range(Elem.shape[0]):
            temp.append(Nind[Elem[i, :]] + 1)
        Elem = np.stack(temp, axis=0)
        connect = Elem.T.flatten()

        DL.main.argtypes = [c_char_p, c_char_p, c_char_p, c_char_p, c_int, c_int, c_int,
                            nc.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
                            nc.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")]
        DL.main(Title,VariableNames, FileName, ScratchDir, Node.shape[-1], Elem.shape[-1], valnum, connect, data_out)

        print('\t\t\t耗时 {:.2f} 秒'.format(time.time() - t1))


    def Get_Stress(self):

        print("单元应力结果计算")
        t1 = time.time()

        self.get_NdSol(self.Num_Elem)
        self.get_NdAve(self.Num_Node)
        vector = self.Stress.to_numpy().transpose((1, 0, 2)).reshape((-1, 6))
        tensor = np.zeros((self.Num_Node, 3, 3), dtype=np.float64)
        tensor[:, 0, 0] = vector[:, 0]
        tensor[:, 1, 1] = vector[:, 1]
        tensor[:, 2, 2] = vector[:, 2]
        tensor[:, 0, 1], tensor[:, 1, 0] = vector[:, 3], vector[:, 3]
        tensor[:, 1, 2], tensor[:, 2, 1] = vector[:, 4], vector[:, 4]
        tensor[:, 0, 2], tensor[:, 2, 0] = vector[:, 5], vector[:, 5]
        principle, _ = nl.eigh(tensor)
        principle = principle[:, ::-1]
        # sortindex = (-np.abs(principle)).argsort(axis=-1)
        # # 排序算法有问题
        # for a, i in zip(principle, sortindex):
        #     a = a[i]
        von_mises = np.sqrt(np.square(principle[:, 0]-principle[:, 1]) + np.square(principle[:, 1]-principle[:, 2]) +
                            np.square(principle[:, 2]-principle[:, 1]))[:, np.newaxis]
        stress = np.concatenate((vector, principle, von_mises), axis=-1)

        print('\t\t\t耗时 {:.2f} 秒'.format(time.time() - t1))

        return stress

    def Output_field_ASC(self, file_name, Fields, Field_names):

        print(file_name + "输出为tecplot文本文件")
        t1 = time.time()

        Elem = self.Physics.Elem[-8:]
        Node = self.Physics.Node[-3:]
        Nind = self.Physics.Nind
        data_out = np.concatenate((Node.T, Fields.transpose(0, 2, 1).reshape((Node.shape[-1], -1))), axis=-1)

        d1 = pd.DataFrame(data_out)
        temp = []
        for i in range(Elem.shape[0]):
            temp.append(Nind[Elem[i, :]] + 1)
        Elem = np.stack(temp, axis=0)
        d2 = pd.DataFrame(Elem.T)

        output_file = self.path + file_name + '.dat'

        f = open(output_file, "w")
        f.write("%s\n" % ('TITLE = ' + '"' + self.name + '.data"'))
        f.write("%s" % ('VARIABLES = "X [ m ]","Y [ m ]","Z [m]"'))
        for step in range(Fields.shape[-1]):
            for name in Field_names:
                f.write("%s " % ',"' + str(name) + str(step + 1) + '"')
        f.write("\n")

        f.write("%s\n" % ('ZONE  I=' + str(Node.shape[-1]) + ', J=' + str(Elem.shape[-1])
                          + ', F=FEPOINT, ET=BRICK'))
        f.close()

        d1.to_csv(output_file, index=False, mode='a', float_format="%10.5e", sep=" ", header=False)
        d2.to_csv(output_file, index=False, mode='a', float_format="10d", sep=" ", header=False)

        print('\t\t\t耗时 {:.2f} 秒'.format(time.time() - t1))


    def taichi_init(self):
        ti.init(arch=ti.cpu, default_fp=ti.f64, default_ip=ti.i32)

        """taichi变量声明"""
        self.GS = ti.field(ti.f64, shape=(2,))
        self.RMat = ti.Matrix.field(3, 1, dtype=ti.f64, shape=8)
        self.DMat = ti.Matrix.field(3, 3, dtype=ti.f64, shape=2)
        self.GMat = ti.Matrix.field(3, 3, dtype=ti.f64, shape=2)

        self.Node = ti.field(ti.f64, shape=(3, self.Num_Node))
        self.Elem = ti.field(ti.i32, shape=(8, self.Num_Elem))
        self.Nind = ti.field(ti.i32, shape=(self.Physics.Nind.shape[-1],))
        self.CoElem = ti.field(ti.i32, shape=(self.Num_Node,))

        self.Displaces = ti.field(ti.f64, shape=(3, self.Num_Node))
        self.Stress = ti.Matrix.field(3, 1, dtype=ti.f64, shape=(2, self.Num_Node))

        """可能需要被优化， 内存占用量太大，是刚度矩阵的1.5倍"""
        self.Coord = ti.Matrix.field(3, 8, dtype=ti.f64, shape=self.Num_Elem)
        self.SVec = ti.Matrix.field(3, 1, dtype=ti.f64, shape=(2, 8, self.Num_Elem))

        """taichi变量赋值"""
        self.Node.from_numpy(self.Physics.Node[-3:])
        self.Elem.from_numpy(self.Physics.Elem[-8:])
        self.Nind.from_numpy(self.Physics.Nind)
        self.CoElem.from_numpy(self.Physics.CoElem)
        self.Displaces.from_numpy(self.Solutions)
        self.FEM_CONST()

        """taichi预编译"""
        self.get_NdSol(0)
        self.get_NdAve(0)

    # 计算每个节点在每个单元的应力大小
    @ti.kernel
    def get_NdSol(self, Num_Elem:ti.i32):
        for i in range(Num_Elem):
            self.UnitVectorS(i)
            for j in range(8):# 节点数
                Ns = self.shapefunc_GS(self.GMat[j])
                ind = self.Nind[self.Elem[j, i]]
                for k in ti.static(range(8)):   # 积分点数
                    self.Stress[0, ind] += Ns[k] * self.SVec[0, k, i]
                    self.Stress[1, ind] += Ns[k] * self.SVec[1, k, i]
    # 应力磨平
    @ti.kernel
    def get_NdAve(self, Num_Node:ti.i32):
        for i in range(Num_Node):
            self.Stress[0,i] /= self.CoElem[i]
            self.Stress[1,i] /= self.CoElem[i]


    @ti.func
    def UnitVectorS(self, i): # 先得到高斯积分点应力计算结果
        # S = ti.Matrix.zero(dt=ti.f64, m=2, n=8)
        for j in range(8):  # 第i个高斯积分点
            rtr = self.RMat[j]
            DNr = self.shape_derv(rtr)
            J = self.Coord[i] @ DNr
            DN = DNr @ J.inverse()
            for k in ti.static(range(8)):   # 第j个节点求解结果
                B1, B2 = ti.Matrix.zero(dt=ti.f64, m=3, n=3), ti.Matrix.zero(dt=ti.f64, m=3, n=3)
                B1[0, 0], B1[1, 1], B1[2, 2] = DN[k, 0], DN[k, 1], DN[k, 2]
                B2[0, 0], B2[0, 1] = DN[k, 1], DN[k, 0]
                B2[1, 1], B2[1, 2] = DN[k, 2], DN[k, 1]
                B2[2, 0], B2[2, 2] = DN[k, 2], DN[k, 0]
                Disp = ti.Matrix.zero(dt=ti.f64, m=1, n=3)
                for m in ti.static(range(3)):
                    Disp[m] = self.Displaces[m, self.Nind[self.Elem[k, i]]]
                R1, R2 = self.DMat[0] @ B1 @ Disp, self.DMat[1] @ B2 @ Disp
                self.SVec[0, j, i] += R1
                self.SVec[1, j, i] += R2


    @ti.func
    def shapefunc_GS(self,rtr):
        Ns = ti.Matrix.zero(dt=ti.f64, m=1, n=8)
        Ns[0] = 0.125 * (1 + rtr[0]/self.GS[1]) * (1 + rtr[1]/self.GS[1]) * (1 - rtr[2]/self.GS[1])
        Ns[1] = 0.125 * (1 - rtr[0]/self.GS[1]) * (1 + rtr[1]/self.GS[1]) * (1 - rtr[2]/self.GS[1])
        Ns[2] = 0.125 * (1 - rtr[0]/self.GS[1]) * (1 - rtr[1]/self.GS[1]) * (1 - rtr[2]/self.GS[1])
        Ns[3] = 0.125 * (1 + rtr[0]/self.GS[1]) * (1 - rtr[1]/self.GS[1]) * (1 - rtr[2]/self.GS[1])
        Ns[4] = 0.125 * (1 + rtr[0]/self.GS[1]) * (1 + rtr[1]/self.GS[1]) * (1 + rtr[2]/self.GS[1])
        Ns[5] = 0.125 * (1 - rtr[0]/self.GS[1]) * (1 + rtr[1]/self.GS[1]) * (1 + rtr[2]/self.GS[1])
        Ns[6] = 0.125 * (1 - rtr[0]/self.GS[1]) * (1 - rtr[1]/self.GS[1]) * (1 + rtr[2]/self.GS[1])
        Ns[7] = 0.125 * (1 + rtr[0]/self.GS[1]) * (1 - rtr[1]/self.GS[1]) * (1 + rtr[2]/self.GS[1])
        return Ns

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
    def shape_derv(self, rtr):  # 形函数对局部坐标偏导
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

        self.GS[0], self.GS[1] = -0.57735026918962, 0.57735026918962
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


        # 确定原节点局部坐标
        self.GMat[0][0], self.GMat[0][1], self.GMat[0][2] = -1, -1, -1
        self.GMat[1][0], self.GMat[1][1], self.GMat[1][2] = -1, -1, 1
        self.GMat[2][0], self.GMat[2][1], self.GMat[2][2] = -1, 1, -1
        self.GMat[3][0], self.GMat[3][1], self.GMat[3][2] = -1, 1, 1
        self.GMat[4][0], self.GMat[4][1], self.GMat[4][2] = 1, -1, -1
        self.GMat[5][0], self.GMat[5][1], self.GMat[5][2] = 1, -1, 1
        self.GMat[6][0], self.GMat[6][1], self.GMat[6][2] = 1, 1, -1
        self.GMat[7][0], self.GMat[7][1], self.GMat[7][2] = 1, 1, 1

        for i in range(self.Num_Elem):
            for j, k in ti.static(ti.ndrange(8, 3)):
                self.Coord[i][k, j] = self.Node[k, self.Nind[self.Elem[j, i]]]