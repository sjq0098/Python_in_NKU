users = [ #用户信息，name、pwd、role分别对应用户名、密码和角色
        dict(name='user1', pwd='123', role='admin'),
        dict(name='user2', pwd='456', role='user'),
        dict(name='user3', pwd='789', role='guest')
    ]
login_user = None #记录登录用户的相关信息
forbiddens = dict( #定义每个角色禁止的操作
        admin=[], #admin可以进行所有操作
        user=['manage_items'], #user不可以做数据项管理操作
        guest=['add_data','del_data','update_data','manage_items'] #guest只能做数据查询操作
    )

def login(): #登录
    global login_user
    name = input('请输入用户名：')
    pwd = input('请输入密码：')
    for u in users:
        if u['name']==name and u['pwd']==pwd: #用户名和密码匹配
            login_user = u
            print('欢迎你，%s'%name)
            return True
    print('用户名或密码不正确！')
    return False            
    
def permission_check(func): #用于权限判断的函数
    def inner(*args, **kwargs): #定义内层函数
        role = login_user['role'] #获取当前登录用户的角色
        if func.__name__ in forbiddens[role]: #在禁止操作列表中
            print('没有该操作的权限！')
            input('按回车继续......')
        else:
            return func(*args, **kwargs) #调用被装饰的函数
    return inner

if __name__=='__main__': #当直接执行该脚本文件时，if条件成立
    pass #在此处可以编写一些测试代码

