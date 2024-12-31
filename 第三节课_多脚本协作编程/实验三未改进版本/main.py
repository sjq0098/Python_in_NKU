import numpy as np
a_x=int(input("请输入要进行乘积的第一个矩阵的行数："))
a_y=int(input("请输入要进行乘积的第一个矩阵的列数："))
b_x=int(input("请输入要进行乘积的第二个矩阵的行数："))
b_y=int(input("请输入要进行乘积的第二个矩阵的列数："))

c_x=int(input("请输入要进行加法的第一个矩阵的行数："))
c_y=int(input("请输入要进行加法的第一个矩阵的列数："))
d_x=int(input("请输入要进行加法的第二个矩阵的行数："))
d_y=int(input("请输入要进行加法的第二个矩阵的列数："))

e_x=int(input("请输入要进行哈达玛积的第一个矩阵的行数："))
e_y=int(input("请输入要进行哈达玛积的第一个矩阵的列数："))
f_x=int(input("请输入要进行哈达玛积的第二个矩阵的行数："))
f_y=int(input("请输入要进行哈达玛积的第二个矩阵的列数："))

g_x=int(input("请输入要进行转置的矩阵的行数："))
g_y=int(input("请输入要进行转置的矩阵的列数："))

matrix1 = np.random.randint(-5, 5, (a_x, a_y))
matrix2 = np.random.randint(-5, 5, (b_x, b_y))
matrix3 = np.random.randint(-5, 5, (c_x, c_y))
matrix4 = np.random.randint(-5, 5, (d_x, d_y))
matrix5 = np.random.randint(-5, 5, (e_x, e_y))
matrix6 = np.random.randint(-5, 5, (f_x, f_y))
matrix7 = np.random.randint(-5, 5, (g_x, g_y))
print(matrix1)
print(matrix2)
print(matrix3)
print(matrix4)
print(matrix5)
print(matrix6)
print(matrix7)
print('\n')


import 实验三_function
实验三_function.martixtimes()
实验三_function.martixplus()
实验三_function.martixhada()
实验三_function.martixTran()