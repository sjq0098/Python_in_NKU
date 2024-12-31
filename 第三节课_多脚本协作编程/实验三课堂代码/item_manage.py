from permission_check import permission_check
@permission_check #加上权限判断的装饰器
def manage_items(iteminfo_list): #定义用于管理数据项的manage_items
    while True: #永真循环
        print('请输入数字进行相应操作：')
        print('1 数据项录入')
        print('2 数据项删除')
        print('3 数据项修改')
        print('4 数据项查询')
        print('0 返回上一层操作')
        subop = int(input('请输入要进行的操作（0~4）：'))
        if subop<0 or subop>4: #输入的操作不存在
            print('该操作不存在，请重新输入！')
            continue
        elif subop==0: #返回上一层操作
            return #结束manage_items函数执行
        elif subop==1: #数据项录入
            add_item(iteminfo_list) #调用add_item函数实现数据项录入
        elif subop==2: #数据项删除
            del_item(iteminfo_list) #调用del_item函数实现数据项删除
        elif subop==3: #数据项修改
            update_item(iteminfo_list) #调用update_item函数实现数据项修改
        else: #数据项查询
            query_item(iteminfo_list) #调用query_item函数实现数据项查询
        input('按回车继续......')

@permission_check #加上权限判断的装饰器
def input_item_type(): #定义用于输入数据项类型的input_item_type函数
    ls_dtype = ['字符串', '整数', '实数'] #支持的数据类型列表
    while True: #永真循环
        dtype = int(input('请输入数据项数据类型（0 字符串，1 整数，2 实数）：'))
        if dtype<0 or dtype>2:
            print('输入的数据类型不存在，请重新输入！')
            continue
        break
    return ls_dtype[dtype]

@permission_check #加上权限判断的装饰器
def add_item(iteminfo_list): #定义实现数据项录入功能的add_item函数
    iteminfo = {} #使用字典保存数据项信息
    iteminfo['name'] = input('请输入数据项名称：')
    for tmp_iteminfo in iteminfo_list: #遍历每一个数据项
        if iteminfo['name']==tmp_iteminfo['name']: #如果该数据项已存在
            print('该数据项已存在！')
            break
    else: #该数据项不存在
        iteminfo['dtype'] = input_item_type() #调用input_item_type函数输入数据项类型
        iteminfo_list += [iteminfo] #将该数据项信息加到iteminfo_list列表的最后
        print('数据项录入成功！')

@permission_check #加上权限判断的装饰器
def del_item(iteminfo_list): #定义实现数据项删除功能的del_item函数
    itemname = input('请输入要删除的数据项名称：')
    for idx in range(len(iteminfo_list)): #遍历每一个数据项的索引
        tmp_iteminfo = iteminfo_list[idx]
        if itemname==tmp_iteminfo['name']: #如果该数据项存在
            del iteminfo_list[idx] #删除该数据项
            print('数据项删除成功！')
            break
    else:
        print('该数据项不存在！')

@permission_check #加上权限判断的装饰器
def update_item(iteminfo_list): #定义实现数据项修改功能的update_item函数
    itemname = input('请输入要修改的数据项名称：')
    for tmp_iteminfo in iteminfo_list: #遍历每一个数据项
        if itemname==tmp_iteminfo['name']: #如果该数据项存在
            tmp_iteminfo['dtype'] = input_item_type() #调用input_item_type函数输入数据项类型
            print('数据项修改成功！')
            break
    else:
        print('该数据项不存在！')

@permission_check #加上权限判断的装饰器
def query_item(iteminfo_list): #定义实现数据项查询功能的query_item函数
    for iteminfo in iteminfo_list: #遍历数据项信息
        print('数据项名称：%s，数据类型：%s'%(iteminfo['name'], iteminfo['dtype']))

if __name__=='__main__': #当直接执行该脚本文件时，if条件成立
    pass #在此处可以编写一些测试代码
