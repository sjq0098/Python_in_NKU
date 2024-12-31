import os
import re
import sys
import signal

def read_user_data(userdata, file_path):
    """从文件中读取用户数据并加载到userdata字典中"""
    try:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                for line in file:
                    username, password = line.strip().split(',')
                    userdata[username] = password
        print("用户数据加载完成")
    except Exception as e:
        print(f"用户数据加载出错: {e}")


def write_user_data(userdata, file_path):
    """将userdata字典中的数据保存到文件"""
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            for username, password in userdata.items():
                file.write(f"{username},{password}\n")
        print(f"正在保存到文件: {file_path}")
        print("用户数据保存成功")
    except Exception as e:
        print(f"用户数据保存出错: {e}")


def delete_user_data(username, userdata, file_path):
    """删除指定用户名的用户数据并保存"""
    try:
        if username in userdata:
            del userdata[username]
            write_user_data(userdata, file_path)
            print("用户数据删除成功")
        else:
            print(f"用户名 {username} 不存在")
    except Exception as e:
        print(f"用户数据删除出错: {e}")


def modify_user_data(username, new_password, userdata, file_path):
    """修改指定用户名的密码并保存"""
    try:
        if username in userdata:
            userdata[username] = new_password
            write_user_data(userdata, file_path)
            print("用户数据修改成功")
        else:
            print(f"用户名 {username} 不存在")
    except Exception as e:
        print(f"用户数据修改出错: {e}")


def if_valid_password(password):
    """验证密码的合法性"""
    try:
        length_pattern = r'.{8,}'
        lowercase_pattern = r'(?=.*[a-z])'
        uppercase_pattern = r'(?=.*[A-Z])'
        digit_pattern = r'(?=.*[0-9])'
        special_pattern = r'(?=.*[@$!%*?&])'
        combine_pattern = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*[0-9])(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$'

        if not re.match(length_pattern, password):
            raise ValueError("密码长度必须大于等于8位！")
        if not re.match(lowercase_pattern, password):
            raise ValueError("密码必须包含至少一个小写字母！")
        if not re.match(uppercase_pattern, password):
            raise ValueError("密码必须包含至少一个大写字母！")
        if not re.match(digit_pattern, password):
            raise ValueError("密码必须包含至少一个数字！")
        if not re.match(special_pattern, password):
            raise ValueError("密码必须包含至少一个特殊字符！")

        if re.match(combine_pattern, password):
            return True
    except ValueError as e:
        print(f"密码验证错误: {e}")
        return False
    except Exception as e:
        print(f"验证密码时出现未知错误: {e}")
        return False


def register_user(users_data, file_path):
    """注册新用户"""
    username = input("请输入用户名: ").strip()
    if username in users_data:
        print("用户名已存在！")
        return
    while True:
        password = input("请输入密码：")
        if if_valid_password(password):
            break  # 密码符合要求，跳出循环
        else:
            print("密码不符合要求，请重新输入！")
    
    users_data[username] = password  # 添加新用户到内存
    write_user_data(users_data, file_path)  # 写入文件
    print("注册成功！")


def login(userdict):
    try:
        input_name = input("请输入用户名：").strip()
        input_password = input("请输入密码：").strip()
        
        if input_name in userdict:
            if userdict[input_name] == input_password:
                print("登录成功！")
                return True  # 返回True表示登录成功
            else:
                print("密码错误！")
                return False  # 密码错误时返回False
        else:
            print("用户名不存在！")
            return False  # 用户名不存在时返回False
    except Exception as e:
        print(f"登录时出现错误: {e}")
        return False  # 如果出现异常，返回False


def create_signal_handler(users_data, file_path):
    """创建信号处理器并传递额外参数"""
    def signal_handler(signum, frame):
        """处理Ctrl+C信号"""
        print("正在保存用户数据")
        write_user_data(users_data, file_path)
        print("程序退出...")
        sys.exit(0)
    return signal_handler
        
            

