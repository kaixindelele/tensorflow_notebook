#coding:utf-8
#导入模块，生成模拟数据集
import numpy as np 
import matplotlib.pyplot as plt 
seed = 2
def generator():
	# 基于seed产生随机数
	rdm = np.random.RandomState(seed)
	# 随机数返回300行2列的矩阵，表示300组坐标点(x0,x1)作为输入数据集
	X = rdm.randn(300,2)
	# 从X这个300行2列的矩阵中取出一行，判断如果两个坐标平方和小于2，给Yield赋值1，其余赋值0
	# Y_作为输入数据集的标签(正确答案)
	Y_ = [int(x0*x0+x1*x1<2) for (x0,x1) in X]
	# 遍历Y中每个元素，1赋值'red',其余赋值'blue',这样便于可视化观察
	Y_c =[['red' if y else 'blue'] for y in Y_]
	# 对数据集X和标签Y进行形状整理，第一个元素为-1表示跟随第二列计算；
	# 第二个元素表示多少列，可见X为两列，Y为1列
	X = np.vstack(X).reshape(-1,2)
	Y_ = np.vstack(Y_).reshape(-1,1)

	return X,Y_,Y_c

if __name__ == '__main__':
    X,Y,Y_c = generator()
    X = np.array(X)
    Y = np.array(Y)
    Y_c = np.array(Y_c)
    print("X:",X.shape)
    print("Y:",Y.shape)
    print("Y_c:",Y_c.shape)
    print("show_over")
