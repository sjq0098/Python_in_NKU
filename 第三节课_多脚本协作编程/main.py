#此脚本用于调用函数和输入输出
import function
# if __name__=="__main__":
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
matrix1 = function.matrix_generation( a_x, a_y)
matrix2 = function.matrix_generation( b_x, b_y)
matrix3 = function.matrix_generation( c_x, c_y)
matrix4 = function.matrix_generation( d_x, d_y)
matrix5 = function.matrix_generation( e_x, e_y)
matrix6 = function.matrix_generation( f_x, f_y)
matrix7 = function.matrix_generation( g_x, g_y)
print(matrix1)
print(matrix2)
print(matrix3)
print(matrix4)
print(matrix5)
print(matrix6)
print(matrix7)
print(" ")
matrix=function.martixtimes(matrix1,matrix2,a_x,a_y,b_x,b_y)
print(matrix)
matrix=function.martixplus(matrix3,matrix4,c_x,c_y,d_x,d_y)
print(matrix)
matrix=function.martixhada(matrix5,matrix6,e_x,e_y,f_x,f_y)
print(matrix)
matrix=function.martixTran(matrix7,g_x,g_y)
print(matrix)


# 3
# 3
# 3
# 3
# 3
# 3
# 3
# 3
# 3
# 3
# 3
# 3
# 3
# 3
#  