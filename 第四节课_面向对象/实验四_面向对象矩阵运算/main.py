from mymartix import mymartix

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

martix1 = mymartix(a_x, a_y)
martix2 = mymartix( b_x, b_y)
martix3 = mymartix( c_x, c_y)
martix4 = mymartix( d_x, d_y)
martix5 = mymartix( e_x, e_y)
martix6 = mymartix( f_x, f_y)
martix7 = mymartix( g_x, g_y)

mymartix.Print(martix1)
mymartix.Print(martix2)
mymartix.Print(martix3)
mymartix.Print(martix4)
mymartix.Print(martix5)
mymartix.Print(martix6)
mymartix.Print(martix7)
print(" ")
mymartix.Print(mymartix.martixTran(martix7))
mymartix.Print(mymartix.martixtimes(martix1,martix2))
mymartix.Print(mymartix.martixplus(martix3,martix4))
mymartix.Print(mymartix.martixhada(martix5,martix6))
