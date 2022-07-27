# FEMtaich

基于[taichi](https://github.com/yuanming-hu/taichi)语言的三维超大自由度并行有限元方法
目前支持

> 1. **有限元网格读取（ANSYS/ANSA-cdb文件）**
> 2. **高效的线性有限元矩阵计算及组装；**
> 3. **线性有限元静力学求解；**
> 4. **线性有限元模态分析；**
> 5. **几何大变形非线性分析；**
> 6. **快速的后处理（tecplot可视化）；**

## Requirements
> pandas==1.1.1
> pymatsolver==0.1.2
> scipy==1.5.2
> numpy==1.19.1
> taichi==0.6.28

## 程序调用格式
1. 指定分析过程名称----name
	* name同时也是网格文件,边界条件文件,载荷文件的前缀;
	* Materials 为字典,指定材料参数,后续补充多种材料;
	* Omegas为全局角速度向量为list或tuple;
``` python
    name = 'long'
    sys.stdout = Logger("Results\\" + name + "\\log.txt")  # 这里将Log输出到指定文件中
    Materials = {'DE': 7.8e-9, 'EX': 200000.0, 'PO': 0.3}  # 材料物性
    Omegas = (0, 0, 100)  # 全局角速度
```
2. 读取网格文件以及边界条件----
	* from READ_physics import Read
	* 采用稀疏矩阵运算节点-单元关系,但读入数据库文件部分还需优化;
``` python
    print('网格文件读取.......')
    t1 = time.time()
    PHYS.mesh()
    print('\t\t\t耗时 {:.2f} 秒'.format(time.time() - t1))

    print('边界条件文件读取.......')
    t1 = time.time()
    PHYS.bounds("bot")  # 边界条件文件前缀，凡是 name_bot前缀文件全部为边界条件节点， 后缀为.lis
    print('\t\t\t耗时 {:.2f} 秒'.format(time.time() - t1))

    print('载荷文件读取.......')
    t1 = time.time()
    PHYS.forces("top")  # 加载信息文件前缀，凡是 name_top前缀文件全部为加载信息节点， 后缀为.lis
    print('\t\t\t耗时 {:.2f} 秒'.format(time.time() - t1))


    print('网格信息预处理.......')
    t1 = time.time()
    PHYS.pre_process()
    print('\t\t\t耗时 {:.2f} 秒'.format(time.time() - t1))

```
3. 进行有限元计算----组装刚度矩阵\质量矩阵\施加载荷\边界条件
	* from FEM_model_csr import Model
	* 本部分完全由taichi进行预编译和计算,后续测试gpu版本;
	> from FEM_model_csr import Model
``` python
        """ 有限元刚度/质量矩阵组装 """
    FEMS = Model(name, PHYS, Materials, Omegas=Omegas)

    print('开始刚度矩阵组装......')
    t1 = time.time()
    K = FEMS.Get_MatrixK()
    print('\t\t\t耗时 {:.2f} 秒'.format(time.time() - t1))

    print('开始质量矩阵组装......')
    t1 = time.time()
    M = FEMS.Get_MatrixM()
    print('\t\t\t耗时 {:.2f} 秒'.format(time.time() - t1))

    print('载荷向量添加......')
    t1 = time.time()
    F = FEMS.Get_VectorF()
    print('\t\t\t耗时 {:.2f} 秒'.format(time.time() - t1))

    ti.reset()
    del FEMS
    gc.collect()

```

4. 指定求解过程----目前给出了线性静态求解和模态分析求解
	* import pymatsolver,使用mkl-pardiso大型稀疏矩阵求解静力学问题;
	* from SOLVE_eigen import Block_lancozs, 编写Block lanczos算法求解对称正定稀疏矩阵特征值;
``` python
    """ 求解静力学"""
    import pymatsolver
    print("Pardiso 开始求解：")
    t1 = time.time()
    K_inv = pymatsolver.Pardiso(K, is_symmetric=True)
    displace = K_inv * F
    print('\t\t\t方程求解误差 {:.2e}'.format(np.linalg.norm(K_inv * F - displace, np.inf)))
    print('\t\t\t耗时 {:.2f} 秒'.format(time.time() - t1))

    """ 求解模态问题"""
    from SOLVE_eigen import Block_lancozs

    cal_fre = 100
    out_fre = 50
    print("Block_lancozs 开始求解：")
    t1 = time.time()
    lanczos = Block_lancozs(K, M, K_inv, num_eigen=cal_fre)  # 求解频率
    frequency, vibration = lanczos.Solve_cpu()
    frequency = (np.sqrt(frequency)) / 2 / np.pi
    for i in range(out_fre):                                # 输出频率个数
        print('\t\t\t第{:3d} 阶固有频率 {:.3e}'.format(i+1, frequency[i]))
    print('\t\t\t耗时 {:.2f} 秒'.format(time.time() - t1))

```

5. 后处理及Tecplot云图结果输出----输出静力学以及模态问题结果: 网格以及物理场
	* from POST_process import Post
	* 最终结果输出在Results//name//文件夹中
``` python
    """ 后处理 """
    Displaces = displace.reshape((PHYS.Num_Node, 3, -1))   # 场的维度包括三个：节点数，变量个数，载荷步或模态数
    POST = Post(name, PHYS, Materials, Displaces)

    print("单元应力结果计算")
    t1 = time.time()
    Stress = POST.Get_Stress()[:, :, np.newaxis]
    print('\t\t\t耗时 {:.2f} 秒'.format(time.time() - t1))

    Fields = np.concatenate((Displaces, Stress), axis=1)
    print("statics结果输出为Tecplot文件")
    t1 = time.time()
    Fields_name = ['u', 'v', 'w', 's-xx', 's-yy', 's-zz', 's-xy', 's-yz', 's-xz', '1-prin', '2-prin', '3-prin', 'von-mises']
    POST.Output_field('statics', Fields, Fields_name)
    print('\t\t\t耗时 {:.2f} 秒'.format(time.time() - t1))

    Fields = vibration[:, :out_fre].reshape((PHYS.Num_Node, 3, -1))   # 只输出out_fre个振型
    print("vibration结果输出为Tecplot文件")
    t1 = time.time()
    Fields_name = ['u', 'v', 'w']
    POST.Output_field('vibration', Fields, Fields_name)
    print('\t\t\t耗时 {:.2f} 秒'.format(time.time() - t1))

```

## 测试网格文件
> 0. 所有测试网格文件上传至[FEM_taichi网格](https://pan.baidu.com/s/1X4Cm7MEvvVRBDq_-TeCsIw) 提取码：k3xr
> 1. 测试网格文件为mesh目录下文件，name为 2，20，50，100，long, 分别为**<font color=red>2万，20万，50万, 100万和200万</font>**的网格节点；
> 2. 相应的边界条件以及载荷文件在boundary下，格式为ANSYS导出的节点集，命名规则为
	> 边界条件文件前缀，凡是 name_bot前缀文件全部为边界条件节点， 后缀为.lis，
	> 加载信息文件前缀，凡是 name_top前缀文件全部为加载信息节点， 后缀为.lis;


## 8.25更新：   
* taichi预编译部分重写，不作为计算时间统计；   
* 所有类重新封装，只输出numpy以及scipy数组和矩阵；   
* 添加计算应力模块，包括主应力、von-mises应力计算。

## 8.29更新:    
* taichi中添加两种计算有限元刚度矩阵和质量矩阵的方法   
	> 1. 原始版本是COO格式； 
	> 2. 添加CSR格式<矩阵计算耗时减少<font color=red>**20%-25%**</font>,<内存占用减少25%>;
	> 3. 添加BSR格式<矩阵计算耗时减少<font color=red>**25%-30%**</font>,<内存占用减少30%>;
	> 4. 推荐使用CSR格式和BSR格式，后续会去掉COO格式；   
* 修改模态分析的bug,目前频率计算恢复正常；   
* 修改部分变量名和输出文件格式；   

## 8.31更新:   
* 添加了支持之前CUDA有限元软件所使用的网格文件读入；
	> 1. cdb文件为ansa输出的格式；
	> 2. 可以使用将ansys中将node和element分别输出的文件形式
	> 3. 具体形式见测试网格文件中的两种形式； 
* 修改了200万以上网格读取文件的bug，并优化了内存占用；
	> 1. 上一个版本使用了ReNode的密集矩阵存储关联节点； 
	> 2. 本次版本采用ReNode的稀疏矩阵保存关联节点，显著节省了内存;     
* 添加了计算过程的log文件，将所有控制台输出的内容记录在结果路径的txt中；   
* 测试了某<font color=DarkGreen>**230万节点**</font>的整圈长叶片<font color=DarkGreen>**100阶**</font>模态的求解时间（无应力刚化）；   
	> 1. taichi预处理时间约**<font color=Dodgerblue>20s</font>**，ANSYS的预处理时间约<font color=red>**90s**</font> ； 
	> 2. taichi模态求解时间(100阶)约**<font color=Dodgerblue>10min</font>**，ANSYS求解时间**<font color=red>25min</font>**，频率误差3%左右;   

## 9.08更新:  
* 添加了静力学大变形非线性的求解部分，并对照了ANSYS（20万节点），结果正确，但存在以下问题：
	>1. 预编译时间过长，intel i7 9750编译达到<font color=Dodgerblue>**18s**</font>，之前的预编译时间只需<font color=red>**3s**</font>左右；
	>2. 计算时间明显较长，20万节点网格矩阵组装需要<font color=Dodgerblue>**10.5s**</font>左右，而线性矩阵组装只需<font color=red>**1.5s**</font>；
	>3. 内存也需要优化，不必要占用内存增长了1倍；
	>4. ANSYS执行相同任务只需<font color=red>**10s**</font>左右；
* 添加了solver的模块，用于将静力学和模态问题求解形式化，方便后续开发；
* 将读取文件和有限元模型建立的部分重新放入class初始化中，简化程序的写法；
* 仍需要改进的点：
	>1. tecplot的输出文件应该为二进制plt格式文件，以及分布式plt文件（tecplot2015版）；
	>2. 模态问题的应力刚化以及旋转软化矩阵；
	>3. 模态减缩方法以及瞬态响应求解；
	>这三点最好在<font color=red>**九月份**</font>完成；
* 需要探索的问题：
	>1. 尽快进行cuda- gpu的单元组装测试；
	>2. 基于GPU的多重网格求解器 paraluation；
	>3. 自动微分方法，为非线性振动求解做铺垫；
## 9.11更新:  
* 添加了Tecplot二进制文件输出函数，调用Tecplot API完成，文件输出加速效果达到<font color=Dodgerblue>**20倍左右**</font>：
  >1. 在20万网格100阶模态结果输出测试结果：ASCII文件<font color=Dodgerblue>**90s**</font>，BINARY文件仅需<font color=Red>**4.75s**</font>;
  >2. 在50万网格100阶模态结果输出测试结果：ASCII文件<font color=Dodgerblue>**240s**</font>，BINARY文件仅需<font color=Red>**13.19s**</font>;
  >3. 后续需要增加szplt格式，据官网说法相对于plt文件更快。
* 重新改写各个功能模块，Lanczos算法的求解器调用方法：
	>1. 刚度/质量矩阵的组装封装在求解过程中；
	>2. 时间记录封装在各模块内部，将外部程序简洁化；
	>3. 新的线性有限元程序调用格式：
``` python
	name = '50'

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

    Fields = Vibrations.reshape((PHYS.Num_Node, 3, -1))
    Fields_name = ['u', 'v', 'w']
    POST.Output_field_ASC('vibration', Fields, Fields_name)
```

<img src="C:\Users\shao\AppData\Roaming\Typora\typora-user-images\image-20200911174039262.png" alt="image-20200911174039262" style="zoom:25%;" /><img src="C:\Users\shao\AppData\Roaming\Typora\typora-user-images\image-20200911174334063.png" alt="image-20200911174334063" style="zoom:25%;" /><img src="C:\Users\shao\AppData\Roaming\Typora\typora-user-images\image-20200911174436247.png" alt="image-20200911174436247" style="zoom:25%;" />

<img src="C:\Users\shao\AppData\Roaming\Typora\typora-user-images\image-20200911174809827.png" alt="image-20200911174809827" style="zoom:25%;" /><img src="C:\Users\shao\AppData\Roaming\Typora\typora-user-images\image-20200911174829209.png" alt="image-20200911174829209" style="zoom:25%;" /><img src="C:\Users\shao\AppData\Roaming\Typora\typora-user-images\image-20200911174846324.png" alt="image-20200911174846324" style="zoom:25%;" />

<img src="C:\Users\shao\AppData\Roaming\Typora\typora-user-images\image-20200911174509467.png" alt="image-20200911174509467" style="zoom:25%;" /><img src="C:\Users\shao\AppData\Roaming\Typora\typora-user-images\image-20200911174540039.png" style="zoom:25%;" /><img src="C:\Users\shao\AppData\Roaming\Typora\typora-user-images\image-20200911174610770.png" alt="image-20200911174610770" style="zoom:25%;" />