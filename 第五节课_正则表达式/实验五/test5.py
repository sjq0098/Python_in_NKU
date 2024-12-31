import re
userdict={'sample_name':{'email':'email@163.com','password':'PassWord123@'},}
def is_valid_email(email):
    pattern=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.]+\.[a-zA-Z]{2,}$'
    if re.match(pattern,email):
        return True
    else:
        return False

def if_valid_password(password):
    erro_message=""
    lenth_pattern=r'.{8,}'
    lowercase_pattern=r'(?=.*[a-z])'
    uppercase_pattern=r'(?=.*[A-Z])'
    digit_pattern=r'(?=.*[0-9])'
    special_pattern=r'(?=.*[@$!%*?&])'
    combine_pattern=r'^(?=.*[a-z])(?=.*[A-Z])(?=.*[0-9])(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$'    
    if re.match(combine_pattern,password):
        return True,erro_message
    else:
        if not re.match(lenth_pattern,password):
            erro_message+="密码长度必须大于等于8位！"+"\n"
        if not re.match(lowercase_pattern,password):
            erro_message+="密码必须包含至少一个小写字母！"+"\n"
        if not re.match(uppercase_pattern,password):
            erro_message+="密码必须包含至少一个大写字母！"+"\n"
        if not re.match(digit_pattern,password):
            erro_message+="密码必须包含至少一个数字！"+"\n"
        if not re.match(special_pattern,password):
            erro_message+="密码必须包含至少一个特殊字符！"+"\n"
        return False,erro_message

def register():
    input_name=input("请输入用户名：")
    while True:
        if input_name in userdict:
            input_name=input("用户名已存在，请重新输入！")
            
        else:
            break
    email=input("请输入邮箱：")
    while True:
        if not is_valid_email(email):
            email=input("邮箱格式不正确，请重新输入：")
        else:
            break
    password=input("请输入密码：")
    while True:
        password_valid,message=if_valid_password(password)
        if not password_valid:
            password=input(message+"请重新输入：")
        else:
            break
    userdict[input_name]={"email":email,"password":password}
    print("注册成功！")

while True:
    print("正在进行注册")
    register()
    choice=input("是否继续注册？(y/n)")
    if choice=='n':
        break
    else:
        continue

