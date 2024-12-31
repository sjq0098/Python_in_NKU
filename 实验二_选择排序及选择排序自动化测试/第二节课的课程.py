ls_data = [] #使用该列表保存所有数据
ls_iteminfo = [] #使用该列表保存每一条数据所包含的数据项信息（名称和数据类型）
while True: #永真循环
    print('请输入数字进行相应操作：')
    print('1 数据录入')
    print('2 数据删除')
    print('3 数据修改')
    print('4 数据查询')
    print('5 数据项维护')
    print('0 退出程序')
    op = int(input('请输入要进行的操作（0~5）：'))
    if op<0 or op>5: #输入的操作不存在
        print('该操作不存在，请重新输入！')
        continue
    elif op==0: #退出程序
        break #结束循环
    elif op==1: #数据录入
        if len(ls_iteminfo)==0: #如果没有数据项
            print('请先进行数据项维护！')
        else:
            data = {} #每条数据用一个字典保存
            for iteminfo in ls_iteminfo: #遍历每一个数据项信息
                itemname = iteminfo['name'] #获取数据项名称
                value = input('请输入%s：'%itemname) #输入数据项值
                #根据数据项数据类型将输入字符串转为整数或实数
                if iteminfo['dtype'] == '整数':
                    value = int(value)
                elif iteminfo['dtype'] == '实数':
                    value = eval(value)
                data[itemname] = value #将数据项保存到data中
            ls_data += [data] #将该条数据加到ls_data列表的最后
            print('数据录入成功！')
    elif op==2: #数据删除
        idx = int(input('请输入要删除的数据编号：'))-1
        if idx<0 or idx>=len(ls_data): #如果超出了有效索引范围
            print('要删除的数据不存在！')
        else:
            del ls_data[idx]
            print('数据删除成功！')
    elif op==3: #数据修改
        idx = int(input('请输入要修改的数据编号：'))-1
        if idx<0 or idx>=len(ls_data): #如果超出了有效索引范围
            print('要修改的数据不存在！')
        else:
            data = {} #每条数据用一个字典保存
            for iteminfo in ls_iteminfo: #遍历每一个数据项信息
                itemname = iteminfo['name'] #获取数据项名称
                value = input('请输入%s：'%itemname) #输入数据项值
                #根据数据项数据类型将输入字符串转为整数或实数
                if iteminfo['dtype'] == '整数':
                    value = int(value)
                elif iteminfo['dtype'] == '实数':
                    value = eval(value)
                data[itemname] = value #将数据项保存到data中
            ls_data[idx] = data
            print('数据修改成功！')
    elif op==4: #数据查询
        for idx in range(len(ls_data)): #依次获取每条数据的索引
            print('第%d条数据：'%(idx+1))
            for iteminfo in ls_iteminfo: #遍历每一个数据项信息
                itemname = iteminfo['name'] #获取数据项的名称
                if itemname in ls_data[idx]: #如果存在该数据项
                    print(itemname, '：', ls_data[idx][itemname]) #输出数据项名称及对应的值
                else: #否则，不存在该数据项
                    print(itemname, '：无数据') #输出提示信息
    else: #数据项维护
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
                break #结束循环
            elif subop==1: #数据项录入
                iteminfo = {} #使用字典保存数据项信息
                iteminfo['name'] = input('请输入数据项名称：')
                for tmp_iteminfo in ls_iteminfo: #遍历每一个数据项
                    if iteminfo['name']==tmp_iteminfo['name']: #如果该数据项已存在
                        print('该数据项已存在！')
                        break
                else: #该数据项不存在
                    ls_dtype = ['字符串', '整数', '实数'] #支持的数据类型列表
                    while True: #永真循环
                        dtype = int(input('请输入数据项数据类型（0 字符串，1 整数，2 实数）：'))
                        if dtype<0 or dtype>2:
                            print('输入的数据类型不存在，请重新输入！')
                            continue
                        iteminfo['dtype'] = ls_dtype[dtype]
                        ls_iteminfo += [iteminfo] #将该数据项信息加到ls_iteminfo列表的最后
                        print('数据项录入成功！')
                        break
            elif subop==2: #数据项删除
                itemname = input('请输入要删除的数据项名称：')
                for idx in range(len(ls_iteminfo)): #遍历每一个数据项的索引
                    tmp_iteminfo = ls_iteminfo[idx]
                    if itemname==tmp_iteminfo['name']: #如果该数据项存在
                        del ls_iteminfo[idx] #删除该数据项
                        print('数据项删除成功！')
                        break
                else:
                    print('该数据项不存在！')
            elif subop==3: #数据项修改
                itemname = input('请输入要修改的数据项名称：')
                for tmp_iteminfo in ls_iteminfo: #遍历每一个数据项
                    if itemname==tmp_iteminfo['name']: #如果该数据项存在
                        while True: #永真循环
                            dtype = int(input('请输入数据项数据类型（0 字符串，1 整数，2 实数）：'))
                            if dtype<0 or dtype>2:
                                print('输入的数据类型不存在，请重新输入！')
                                continue
                            tmp_iteminfo['dtype'] = ls_dtype[dtype] #修改数据项数据类型
                            print('数据项修改成功！')
                            break
                        break
                else:
                    print('该数据项不存在！')
            else: #数据项查询
                for iteminfo in ls_iteminfo: #遍历数据项信息
                    print('数据项名称：%s，数据类型：%s'%(iteminfo['name'], iteminfo['dtype']))
            input('按回车继续......')
        continue #通过返回上一层结束循环时，不需要按回车继续
    input('按回车继续......')