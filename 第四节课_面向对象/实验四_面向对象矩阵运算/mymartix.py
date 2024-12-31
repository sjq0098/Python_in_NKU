import numpy as np
class mymartix:
    x=0
    y=0
    martix=None
    def __init__(self,A,B) :
        self.x=A
        self.y=B
        self.martix = np.random.randint(-5, 5, (A, B))

    def martixTran(self):
        if (self.x==0)or(self.y==0):
            print("Wrong")
            return
        result=mymartix(self.y,self.x)
        for i in range(self.y):
            for j in range(self.x):
                result.martix[i][j]=self.martix[j][i]
        return result
    
    def Print(self):
        print(self.martix)

     
    @classmethod 
    def martixtimes(cls,one,theother):
        if(one.y!=theother.x):
            print("Wrong")
            return
        result=mymartix(one.x,theother.y)
        for i in range(one.x):
            for j in range(theother.y):
                for k in range(one.y):
                    result.martix[i][j]+=one.martix[i][k]*theother.martix[k][j]
        return result   
    @classmethod 
    def martixplus(cls,one,theother):
        if(one.x!=theother.x)or(one.y!=theother.y):
            print("Wrong")
            return
        result=mymartix(one.x,theother.y)
        for i in range(one.x):
            for j in range(theother.y):
                result.martix[i][j]+=one.martix[i][j]+theother.martix[i][j]
        return result   
    @classmethod 
    def martixhada(cls,one,theother):
        if(one.x!=theother.x)or(one.y!=theother.y):
            print("Wrong")
            return
        result=mymartix(one.x,theother.y)
        for i in range(one.x):
            for j in range(theother.y):
                result.martix[i][j]+=one.martix[i][j]*theother.martix[i][j]
        return result   
   