#本脚本用于实现多通道多卷积核的卷积操作
import numpy as np
import torch
n =int(input("输入通道数"))
ix,iy=int(input("输入矩阵行数")),int(input("输入矩阵列数"))
m=int(input("请输入卷积核数量"))
kx,ky=int(input("输入卷积核矩阵行数")),int(input("输入卷积核矩阵列数"))

Img=np.random.randint(-5,5,(n,ix,iy))
kernels=np.random.random((m,n,kx,ky))


print(Img)
print(kernels)
print('\n')
result1=np.zeros((m,ix-kx+1,iy-ky+1))
result=np.zeros((m,ix-kx+1,iy-ky+1))

for i in range(m):
    for j in range(n):
        
        for row in range(ix-kx+1):
            for col in range(iy-ky+1):
                result1[i,row,col]+=np.sum(Img[row:row+kx,col:col+ky]*kernels[i,j])
print(result1)
print(result)
