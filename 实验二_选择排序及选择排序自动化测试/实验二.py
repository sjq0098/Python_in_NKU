l=[]
print("请输入数据,输入NULL时结束")
while True:
    x=input()
    if x=="NULL":
        break
    l.append(x)
s=set(l)
l=list(s)
for i in range(0,len(l)-1):
    min_1=l[i]
    min_2=i
    for j in range(i+1,len(l)):
        if min_1>l[j]:
            min_1=l[j]
            min_2=j
    temp=l[i]
    l[i]=l[min_2]
    l[min_2]=temp
print(l)