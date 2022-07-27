import scipy.linalg as sc
import numpy as np
import scipy.sparse as sp
# import pymatsolver
from pyMKL import pardisoSolver
import time


class Eigen_solver:
    def __init__(self, Model, Qtype=0):
        self.Qtype = Qtype   # 取1时求解大变形非线性静力学问题
        self.Model = Model

    def Solve_cpu(self, num_eigen=100, num_block=10):

        print("Block_lancozs 开始求解：")

        t1 = time.time()

        if not hasattr(self.Model, 'KL'):
            t1 = time.time()
            self.Model.KL = self.Model.Get_MatrixKL()
            print('\t\t\t组装耗时 {:.2f} 秒'.format(time.time() - t1))

        if not hasattr(self.Model, 'M'):
            t1 = time.time()
            self.Model.M = self.Model.Get_MatrixM()
            print('\t\t\t组装耗时 {:.2f} 秒'.format(time.time() - t1))

        if not hasattr(self.Model, 'linear_solver'):
            # self.Model.K_inv = pymatsolver.Pardiso(self.Model.KL, is_symmetric=True)
            self.Model.linear_solver = pardisoSolver(self.Model.KT, mtype=2)
            self.Model.linear_solver.factor()

        freq, vect= self.Lanczos_solve(self.Model.KL, self.Model.M, self.Model, num_eigen, num_block)

        freq = (np.sqrt(freq)) / 2 / np.pi

        for i in range(num_eigen):                # 输出频率个数
            print('\t\t\t第{:3d} 阶固有频率 {:.3e}'.format(i+1, freq[i]))
        print('\t\t\t耗时 {:.2f} 秒'.format(time.time() - t1))

        return freq, vect

    def Lanczos_solve(self, A, B, A_, num_eigen=100, num_block=10):

        m = num_eigen * 2
        n = A.shape[0]
        b = int(num_block)

        Q = np.zeros((n, m), dtype=np.float64)
        T = np.zeros((m, m), dtype=np.float64)
        u = np.random.uniform(size=(n, b))
        # v = np.zeros((n, b))

        v = sc.cholesky(u.T@(B@u))
        u = u @ sc.inv(v)

        Q[:, :b] = u
        # while(1):
        for i in range(int(m/b)):

            sta = time.time()
            # v = A_ * (B @ u)
            v = self.Model.linear_solver.solve(B@u)
            T[i*b:(i+1)*b, i*b:(i+1)*b] = u.T @ (B @ v)
            v = v - u @ T[i*b:(i+1)*b, i*b:(i+1)*b]
            if i >= int(m/b) - 1: break
            # for j in range(i+1):
            #     v = v - Q[:, j*b:j*b+b] @ (Q[:, j*b:j*b+b].T @ (B @ v))  修正正交化
            v = v - Q[:, :i*b+b] @ (Q[:, :i*b+b].T @ (B @ v))
            v = v - Q[:, :i * b + b] @ (Q[:, :i * b + b].T @ (B @ v))

            t = sc.cholesky(v.T @ (B @ v))
            T[(i+1)*b:(i+2)*b, i*b:(i+1)*b] = t
            T[i*b:(i+1)*b, (i+1)*b:(i+2)*b] = t.T
            u = v @ sc.inv(t)
            # v = -t * T[i*b:(i+1)*b, (i+1)*b:(i+2)*b]
            Q[:, (i+1)*b:(i+2)*b] = u
            end = time.time()
            print("\t\t\tlanczos_step {:d}, also need {:d}, cost {:.2f}".format(i + 1, int(m / b) - i - 1, end - sta))

        self.Model.linear_solver.clear()
        Eval, Vect = sc.eigh(T)

        Eval = np.ones_like(Eval) / Eval
        Q = Q @ Vect
        ind = np.argsort(Eval)
        return Eval[ind][:num_eigen], Q[:, ind][:, :num_eigen]



if __name__ == "__main__":

    import time

    name = "50"
    t1 = time.time()


    N = 1000
    A = np.zeros((N, N), dtype=np.float64)
    B = np.zeros((N, N), dtype=np.float64)

    for i in range(N):
        A[i, i] = i + 2
        B[i, i] = i + 1
    for i in range(1, N):
        A[i, i-1] = -np.float(i+1)/2
        A[i-1, i] = -np.float(i+1)/2
        B[i, i-1] = np.float(i+1)/2
        B[i-1, i] = np.float(i+1)/2

    SA = sp.lil_matrix((N, N), dtype=np.float64)
    SA[:, :] = A
    SB = sp.lil_matrix((N, N), dtype=np.float64)
    SB[:, :] = B

    SA = SA.tocsr()
    SB = SB.tocsr()

    print('广义特征值请求解.......')
    solver = Eigen_solver(SA, SB)
    Eval, Vect = solver.Solve_cpu()
    print('\t\t\t耗时 {:.2f} 秒'.format(time.time() - t1))


    # eigs, vecs = sl.eigsh(SA, 100, SB, which='SM',)

    c = 0




