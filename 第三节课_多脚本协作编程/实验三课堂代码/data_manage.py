from permission_check import *
from item_manage import manage_items
@permission_check #加上权限判断的装饰器
def manage_data(data_list, iteminfo_list): #定义用于管理数据的manage_data函数
    while True: #永真循环
        print('请输入数字进行相应操作：')
        print('1 数据录入')
        print('2 数据删除')
        print('3 数据修改')
        print('4 数据查询')
        print('5 数据项维护')
        print('6 切换用户')
        print('0 退出程序')
        op = int(input('请输入要进行的操作（0~6）：'))
        if op<0 or op>6: #输入的操作不存在
            print('该操作不存在，请重新输入！')
            continue
        elif op==0: #退出程序
            break #结束循环
        elif op==1: #数据录入
            if len(iteminfo_list)==0: #如果没有数据项
                print('请先进行数据项维护！')
            else:
                add_data(data_list, iteminfo_list)
        elif op==2: #数据删除
            del_data(data_list)
        elif op==3: #数据修改
            update_data(data_list, iteminfo_list)
        elif op==4: #数据查询
            query_data(data_list, iteminfo_list)
        elif op==5: #数据项维护
            manage_items(iteminfo_list)
            continue
        elif op==6: #切换用户
            login()
        input('按回车继续......')

@permission_check #加上权限判断的装饰器
def input_data(iteminfo_list): #定义用于输入一条新数据的input_data函数
    data = {} #每条数据用一个字典保存
    for iteminfo in iteminfo_list: #遍历每一个数据项信息
        itemname = iteminfo['name'] #获取数据项名称
        value = input('请输入%s：'%itemname) #输入数据项值
        #根据数据项数据类型将输入字符串转为整数或实数
        if iteminfo['dtype']=='整数':
            value = int(value)
        elif iteminfo['dtype']=='实数':
            value = eval(value)
        data[itemname] = value #将数据项保存到data中
    return data #将输入的数据返回

@permission_check #加上权限判断的装饰器
def add_data(data_list, iteminfo_list): #定义实现数据录入功能的add_data函数
    data = input_data(iteminfo_list) #调用input_data函数实现数据录入
    data_list += [data] #将该条数据加到data_list列表的最后
    print('数据录入成功！')

@permission_check #加上权限判断的装饰器
def del_data(data_list): #定义实现数据删除功能的del_data函数
    idx = int(input('请输入要删除的数据编号：'))-1
    if idx<0 or idx>=len(data_list): #如果超出了有效索引范围
        print('要删除的数据不存在！')
    else:
        del data_list[idx]
        print('数据删除成功！')

@permission_check #加上权限判断的装饰器
def update_data(data_list, iteminfo_list): #定义实现数据修改功能的update_data函数
    idx = int(input('请输入要修改的数据编号：'))-1
    if idx<0 or idx>=len(data_list): #如果超出了有效索引范围
        print('要修改的数据不存在！')
    else:
        data = input_data(iteminfo_list) #调用input_data函数实现数据录入
        data_list[idx] = data #用该条数据替换data_list中索引值为idx的元素
        print('数据修改成功！')

@permission_check #加上权限判断的装饰器
def query_data(data_list, iteminfo_list): #定义实现数据查询功能的query_data函数
    while True:
        print('请输入数字进行相应查询操作：')
        print('1 全部显示')
        print('2 按数据项查询')
        print('0 返回上一层')
        subop = int(input('请输入要进行的操作（0~2）：'))
        if subop==0: #返回上一层
            break
        elif subop==1: #全部显示
            retTrue = lambda *args,**kwargs:True #定义一个可以接收任何参数并返回True的匿名函数
            show_query_result(data_list, iteminfo_list, retTrue, None) #调用函数显示全部数据
        elif subop==2: #按数据项查询
            condition = {}
            condition['itemname'] = input('请输入数据项名称：')
            for iteminfo in iteminfo_list: #遍历数据项信息
                if iteminfo['name']==condition['itemname']: #如果有匹配的数据项
                    condition['lowval'] = input('请输入最小值：')
                    condition['highval'] = input('请输入最大值：')
                    if iteminfo['dtype']!='字符串': #不是字符串类型，则转换为数值
                        condition['lowval'] = eval(condition['lowval'])
                        condition['highval'] = eval(condition['highval'])
                    show_query_result(data_list, iteminfo_list, judge_condition, condition) #调用函数将满足条件的数据输出
                    break
            else:
                print('该数据项不存在！')
        input('按回车继续......')

def judge_condition(data, condition): #判断data是否满足condition中设置的条件
    itemname = condition['itemname']
    lowval = condition['lowval']
    highval = condition['highval']
    if data[itemname]>=lowval and data[itemname]<=highval:
        return True
    return False
    
def show_query_result(data_list, iteminfo_list, filter_fn, condition): #用于显示查询结果的高阶函数
    for idx in range(len(data_list)): #依次获取每条数据的索引
        if filter_fn(data_list[idx], condition)==False: #如果不满足查询条件
            continue
        print('第%d条数据：'%(idx+1))
        for iteminfo in iteminfo_list: #遍历每一个数据项信息
            itemname = iteminfo['name'] #获取数据项的名称
            if itemname in data_list[idx]: #如果存在该数据项
                print(itemname, '：', data_list[idx][itemname]) #输出数据项名称及对应的值
            else: #否则，不存在该数据项
                print(itemname, '：无数据') #输出提示信息

if __name__=='__main__': #当直接执行该脚本文件时，if条件成立
    from permission_check import login
    if login()==True: #登录（用户名：user1，密码：123）
        ls_item = [ #每条数据包含3个数据项
                dict(name='身高（m）',dtype='实数'),
                dict(name='体重（kg）',dtype='整数'),
                dict(name='交通方式',dtype='字符串'),
            ]
        ls_data = [ #初始填入5条数据
                {'身高（m）':1.62, '体重（kg）':64, '交通方式':'公共交通'},
                {'身高（m）':1.52, '体重（kg）':56, '交通方式':'公共交通'},
                {'身高（m）':1.8, '体重（kg）':77, '交通方式':'公共交通'},
                {'身高（m）':1.8, '体重（kg）':87, '交通方式':'步行'},
                {'身高（m）':1.62, '体重（kg）':53, '交通方式':'汽车'}
            ]
        manage_data(ls_data, ls_item)

