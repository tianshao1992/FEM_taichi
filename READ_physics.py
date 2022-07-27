import pandas as pd
import numpy as np
import os
import scipy.sparse as sp
import time

class Read(object):

    def __init__(self, name, bounds_file=('bot', 'lis'), forces_file=('top', 'lis')):

        self.name = name
        self.mesh_path = 'Mesh\\'
        self.bounds_path = 'Boundary\\'
        self.forces_path = 'Boundary\\'
        self.bounds_file = bounds_file
        self.forces_file = forces_file

        print('网格文件读取.......')
        t1 = time.time()
        self.mesh()
        print('\t\t\t耗时 {:.2f} 秒'.format(time.time() - t1))

        print('边界条件文件读取.......')
        t1 = time.time()
        self.bounds()  # 边界条件文件前缀，凡是 name_bot前缀文件全部为边界条件节点， 后缀为.lis
        print('\t\t\t耗时 {:.2f} 秒'.format(time.time() - t1))

        print('载荷文件读取.......')
        t1 = time.time()
        self.forces()  # 加载信息文件前缀，凡是 name_top前缀文件全部为加载信息节点， 后缀为.lis
        print('\t\t\t耗时 {:.2f} 秒'.format(time.time() - t1))

        print('网格信息预处理.......')
        t1 = time.time()
        self.pre_process()
        print('\t\t\t耗时 {:.2f} 秒'.format(time.time() - t1))

        print("材料参数---\n\t\t\t节点：{:d},单元：{:d},自由度：{:d} ".format(self.Num_Node, self.Num_Elem, self.Num_Node * 3))
        print("\t\t\t最大节点编号：{:d}, 最大波前值：{:d} ".format(self.Num_MaxN, self.CoNode.max()))


    def pre_process(self):

        self.Num_Node = self.Node.shape[-1]
        self.Num_Elem = self.Elem.shape[-1]
        self.Num_MaxN = int(self.Node[0, :].max())

        self.Nind = np.zeros((self.Num_MaxN,), dtype=np.int)
        self.Nind[self.Node[0, :].astype(np.int)-1] = np.linspace(0, self.Num_Node-1, num=self.Num_Node, dtype=np.int)

        Row_Elem = np.linspace(0, self.Num_Elem-1, num=self.Num_Elem, dtype=np.int).repeat(8)
        Col_Node = self.Nind[self.Elem[-8:].T.flatten()]
        Val_Rela = np.ones_like(Row_Elem)

        Elem_Node = sp.coo_matrix((Val_Rela, (Row_Elem, Col_Node)), shape=(self.Num_Elem, self.Num_Node)).tocsc()
        Node_Node = Elem_Node.T * Elem_Node
        Node_Node.sort_indices()
        self.SmNode = Node_Node.indptr
        self.CoNode = Node_Node.indptr[1:] - Node_Node.indptr[:-1]
        self.CoElem = Elem_Node.indptr[1:] - Elem_Node.indptr[:-1]
        self.ReNode = Node_Node.indices


    def mesh(self, ext='.cdb'):

        if os.path.exists(self.mesh_path + str(self.name)+ ext):

            data = pd.read_csv(self.mesh_path + str(self.name) + ext, sep='\t')
            name = data.columns.values.tolist()[0]
            N_STA = data[name].str.contains('NBLOCK')
            N_STA = N_STA[N_STA.values == True].index.tolist()[0]+2
            N_END = data[name].str.contains('N,R5')
            N_END = N_END[N_END.values == True].index.tolist()[0]
            E_STA = data[name].str.contains('EBLOCK')
            E_STA = E_STA[E_STA.values == True].index.tolist()[0]+2
            E_END = data[name].str.strip().str.match('-1')
            E_END = E_END[E_END.values == True].index.tolist()[0]

            Node = data[name].iloc[N_STA:N_END].str.replace('-', ' -').str.replace('E -', 'E-')
            Node = " ".join(Node.tolist())
            Node = " ".join(Node.split()).split()
            Node = np.array(Node, dtype=np.float64).reshape((-1, 6))
            print("读取网格节点数： {:d}".format(Node.shape[0]))

            Elem = " ".join(data[name].iloc[E_STA:E_END].tolist())
            Elem = " ".join(Elem.split()).split()
            Elem = np.array(Elem, dtype=np.int).reshape((-1, 18))
            print("读取网格单元数： {:d}".format(Elem.shape[0]))

        else:
            data = pd.read_csv(self.mesh_path + str(self.name) + "_node.txt", sep="\s+",header=None, skiprows=1)
            Node = data.values.astype(np.float64)
            data = pd.read_csv(self.mesh_path + str(self.name) + "_element.txt", sep="\s+", header=None, skiprows=1)
            Elem = data.values.astype(np.int)

        self.Elem = Elem.T - 1
        self.Node = Node.T

    def bounds(self):

        prefix = self.bounds_file[0]
        ext = self.bounds_file[1]
        files = self.files_get(self.bounds_path, ext)
        Bounds = []

        for file in files:
            if self.name + "_" + prefix in file:
                data = pd.read_csv(file, sep='\t')
                name = data.columns.values.tolist()[0]
                index = data[name].str.contains('NAME|NODE|LIST')
                bound = data[name][index.values==False]
                bound = " ".join(bound.tolist())
                bound = " ".join(bound.split()).split()
                Bounds.append([np.array(bound, dtype=np.int)-1, [0, 0, 0]])
                print("边界条件文件：  ", str(file))

        self.Bounds = Bounds

    def forces(self):

        prefix = self.forces_file[0]
        ext = self.forces_file[1]
        files = self.files_get(self.forces_path, ext)
        Forces = []

        for file in files:
            if self.name + "_" + prefix in file:
                data = pd.read_csv(file, sep='\t')
                name = data.columns.values.tolist()[0]
                index = data[name].str.contains('NAME|NODE|LIST')
                force = data[name][index.values==False]
                force = " ".join(force.tolist())
                force = " ".join(force.split()).split()
                Forces.append([np.array(force, dtype=np.int)-1, [0, 0, 0]])
                print("载荷文件：  ", str(file))

        self.Forces = Forces



    def files_get(self, path, ext=None):

        allfiles = []
        needExtFilter = (ext!=None)

        for root, dirs, files in os.walk(path):
            for filespath in files:
                filepath = os.path.join(root, filespath)
                extension = os.path.splitext(filepath)[1][1:]
                if needExtFilter and extension in ext:
                    allfiles.append(filepath)
                elif not needExtFilter:
                    allfiles.append(filepath)
            return allfiles


if __name__ == "__main__":

    import time

    name = "50"
    t1 = time.time()
    Physics = Read(name)