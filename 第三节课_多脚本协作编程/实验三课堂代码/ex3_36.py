from permission_check import login #从permission_check模块中导入login函数
from data_manage import manage_data #从data_manage模块导入manage_data函数
ls_data = [] #使用该列表保存所有数据
ls_iteminfo = [] #使用该列表保存每一条数据所包含的数据项信息（名称和数据类型）
if login()==True: #登录成功
    manage_data(ls_data, ls_iteminfo) #调用manage_data函数进行数据管理
