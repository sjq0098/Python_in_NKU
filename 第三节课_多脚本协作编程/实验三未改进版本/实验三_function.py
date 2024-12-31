import numpy as np
import main
def martixtimes():
    if main.a_y!=main.b_x:
        print("Wrong")
        return
    martix=np.zeros((main.a_x,main.b_y))
    for i in range(0,main.a_x):
        for j in range(0,main.b_y):
            for k in range(0,main.a_y):
                martix[i][j]+=main.matrix1[i][k]*main.matrix2[k][j]
    print(martix) 
def martixplus():
    if (main.c_x!=main.d_x)or(main.c_y!=main.d_y):
        print("Wrong")
        return
    martix=np.zeros((main.c_x,main.d_y))
    martix=main.matrix3+main.matrix3
    print(martix) 
def martixhada():
    if (main.e_x!=main.f_x)or(main.e_y!=main.f_y):
        print("Wrong")
        return
    martix=np.zeros((main.e_x,main.f_y))
    for i in range(0,main.e_x):
        for j in range(0,main.f_y):
            martix[i][j]=main.matrix5[i][j]*main.matrix6[i][j]
    print(martix) 
def martixTran():
    if (main.g_x==0)or(main.g_y==0):
        print("Wrong")
        return
    martix=np.zeros((main.g_y,main.g_x))
    for i in range(0,main.g_y):
        for j in range(0,main.g_x):
            martix[i][j]=main.matrix7[j][i]
    print(martix)