import numpy as np
import random
def matrix_generation(A, B):  
    L = [[random.randint(-5, 5) for _ in range(B)] for _ in range(A)]  
    return L
def martixtimes(martix1,martix2,a_x,a_y,b_x,b_y):
    if a_y!=b_x:
        print("Wrong")
        return
    martix=np.zeros((a_x,b_y))
    for i in range(a_x):
        for j in range(b_y):
            for k in range(a_y):
                martix[i][j]+=martix1[i][k]*martix2[k][j]
    return martix
    
def martixplus(martix1,martix2,a_x,a_y,b_x,b_y):
    if (a_x!=b_x)or(a_y!=b_y):
        print("Wrong")
        return
    martix=np.zeros((a_x,b_y))
    for i in range(a_x):
        for j in range(b_y):
           martix[i][j]=martix1[i][j]+martix2[i][j]
    return martix
def martixhada(martix1,martix2,a_x,a_y,b_x,b_y):
    if (a_x!=b_x)or(a_y!=b_y):
        print("Wrong")
        return
    martix=np.zeros((a_x,b_y))
    for i in range(a_x):
        for j in range(b_y):
           martix[i][j]=martix1[i][j]*martix2[i][j]
    return martix
def martixTran(martix1,a_x,a_y):
    if (a_x==0)or(a_y==0):
        print("Wrong")
        return
    martix=np.zeros((a_y,a_x))
    for i in range(a_y):
        for j in range(a_x):
            martix[i][j]=martix1[j][i]
    return martix