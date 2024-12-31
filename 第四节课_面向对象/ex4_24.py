from user import User #导入User类
from dataset_manage import DatasetManage #导入DatasetManage类
if User.login()==True: #登录
    DatasetManage.run() #开始进行数据管理操作
