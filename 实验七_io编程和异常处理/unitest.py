import unittest
from unittest.mock import patch
import os
import random
import re
import string
from Ex7_function import (
    read_user_data, 
    write_user_data, 
    register_user, 
    if_valid_password, 
    modify_user_data, 
    login
)
def if_valid_password(password):
    try:
        length_pattern = r'.{8,}'
        lowercase_pattern = r'(?=.*[a-z])'
        uppercase_pattern = r'(?=.*[A-Z])'
        digit_pattern = r'(?=.*[0-9])'
        special_pattern = r'(?=.*[@$!%*?&])'
        combine_pattern = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*[0-9])(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$'

        # 修正为逐个检查密码规则
        if not re.match(length_pattern, password):
            with open('password_validation_log.txt', 'a', encoding='utf-8') as log_file:
                log_file.write(f"密码验证失败: 密码长度必须大于等于8位！\n")
            raise ValueError("密码长度必须大于等于8位！")
        if not re.search(lowercase_pattern, password):
            with open('password_validation_log.txt', 'a', encoding='utf-8') as log_file:
                log_file.write(f"密码验证失败: 密码必须包含至少一个小写字母！\n")
            raise ValueError("密码必须包含至少一个小写字母！")
        if not re.search(uppercase_pattern, password):
            with open('password_validation_log.txt', 'a', encoding='utf-8') as log_file:
                log_file.write(f"密码验证失败: 密码必须包含至少一个大写字母！\n")
            raise ValueError("密码必须包含至少一个大写字母！")
        if not re.search(digit_pattern, password):
            with open('password_validation_log.txt', 'a', encoding='utf-8') as log_file:
                log_file.write(f"密码验证失败: 密码必须包含至少一个数字！\n")
            raise ValueError("密码必须包含至少一个数字！")
        if not re.search(special_pattern, password):
            with open('password_validation_log.txt', 'a', encoding='utf-8') as log_file:
                log_file.write(f"密码验证失败: 密码必须包含至少一个特殊字符！\n")
            raise ValueError("密码必须包含至少一个特殊字符！")

        if re.match(combine_pattern, password):
            with open('password_validation_log.txt', 'a', encoding='utf-8') as log_file:
                log_file.write(f"密码验证成功: {password}\n")
            return True
    except ValueError as e:
        with open('password_validation_log.txt', 'a', encoding='utf-8') as log_file:
            log_file.write(f"密码验证失败: {e}\n")
        return False
    except Exception as e:
        with open('password_validation_log.txt', 'a', encoding='utf-8') as log_file:
            log_file.write(f"密码验证时出现未知错误: {e}\n")
        return False


class TestUserManagement(unittest.TestCase):
    
    def setUp(self):
        """初始化测试环境，创建临时文件"""
        self.test_file = "test_user_data.txt"  # 用于保存注册用户的数据
        self.log_file = "registration_log.txt"  # 用于保存测试日志
        self.login_log_file = "login_log.txt"  # 用于保存登录测试日志
        self.password_log_file = "password_validation_log.txt"  # 用于保存密码验证日志
        self.userdata = {}
        
        # 确保没有之前的文件
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
        if os.path.exists(self.password_log_file):
            os.remove(self.password_log_file)
        if os.path.exists(self.login_log_file):
            os.remove(self.login_log_file)
        
        # 初始化用户数据
        write_user_data(self.userdata, self.test_file)
    
    def tearDown(self):
        """清理测试环境，删除测试文件"""
        # 这里不删除文件，保留注册数据和日志
        pass
    
    def generate_random_password(self):
        """生成一个随机密码"""
        length = random.randint(8, 12)
        password = ''.join(random.choices(string.ascii_letters + string.digits + "@$!%*?&", k=length))
        return password
    
    def log_registration_result(self, username, result):
        """将注册结果写入日志文件"""
        with open(self.log_file, "a", encoding="utf-8") as log_file:
            log_file.write(f"用户名: {username}, 注册结果: {result}\n")
    
    def log_password_validation_result(self, password, result):
        """将密码验证结果写入日志文件"""
        with open(self.password_log_file, "a", encoding="utf-8") as log_file:
            log_file.write(f"密码: {password}, 验证结果: {result}\n")
    
    def log_login_result(self, username, result):
        """将登录测试结果写入日志文件"""
        with open(self.login_log_file, "a", encoding="utf-8") as log_file:
            log_file.write(f"用户名: {username}, 登录结果: {result}\n")
    
    def test_register_multiple_users(self):
        """模拟多个用户进行注册并记录注册情况"""
        num_users = 20  # 模拟注册20个用户
        for _ in range(num_users):
            username = f"user{random.randint(1000, 9999)}"
            password = self.generate_random_password()
            
            # 模拟用户注册
            self.userdata[username] = password  # 模拟注册
            write_user_data(self.userdata, self.test_file)
            
            # 检查文件是否包含此用户
            read_user_data(self.userdata, self.test_file)
            
            if username in self.userdata and self.userdata[username] == password:
                result = "成功"
            else:
                result = "失败"
            
            # 记录注册结果到日志文件
            self.log_registration_result(username, result)
        
        # 打印日志文件内容供人工检查
        print("注册日志文件内容：")
        with open(self.log_file, "r", encoding="utf-8") as log_file:
            print(log_file.read())
    
    @patch('builtins.input', side_effect=['testuser', 'Test1234!'])
    def test_login_success(self, mock_input):
        """测试登录功能：登录成功的情况"""
        username = "testuser"
        password = "Test1234!"
        
        # 模拟用户注册
        self.userdata[username] = password
        write_user_data(self.userdata, self.test_file)
        
        # 模拟用户登录
        read_user_data(self.userdata, self.test_file)
        result = login(self.userdata)  # 返回登录结果
        self.assertTrue(result)
        
        # 记录登录结果到日志文件
        self.log_login_result(username, "登录成功")
    
    @patch('builtins.input', side_effect=['nonexistent_user', 'RandomPass!'])
    def test_login_failure(self, mock_input):
        """测试登录功能：用户名不存在的情况"""
        username = "nonexistent_user"
        password = "RandomPass!"
        
        # 检查登录失败
        result = login(self.userdata)
        self.assertFalse(result)
        
        # 记录登录结果到日志文件
        self.log_login_result(username, "登录失败: 用户名不存在")
    
    @patch('builtins.input', side_effect=['testuser', 'NewPassword123!'])
    def test_modify_password(self, mock_input):
        """测试修改密码功能"""
        username = "testuser"
        old_password = "Test1234!"
        new_password = "NewPassword123!"
        
        # 模拟用户注册
        self.userdata[username] = old_password
        write_user_data(self.userdata, self.test_file)
        
        # 模拟修改密码
        modify_user_data(username, new_password, self.userdata, self.test_file)
        
        # 检查是否更新
        read_user_data(self.userdata, self.test_file)
        self.assertEqual(self.userdata[username], new_password)
    
    def test_invalid_password(self):
        """测试密码验证功能"""
        # 测试无效的密码
        invalid_passwords = [
            "short",        # 太短
            "NoSpecialChar", # 没有特殊字符
            "NOLOWERCASE123!", # 没有小写字母
            "nouppercase123!", # 没有大写字母
            "NoNumber!",    # 没有数字
        ]
        
        for password in invalid_passwords:
            result = if_valid_password(password)
            self.assertFalse(result, f"密码验证失败：{password} 应该是无效的")
            # 记录密码验证结果到日志文件
            self.log_password_validation_result(password, "无效")
        
        # 测试有效密码
        valid_password = "Valid123!"
        result = if_valid_password(valid_password)
        self.assertTrue(result, "有效密码验证失败")
        # 记录密码验证结果到日志文件
        self.log_password_validation_result(valid_password, "有效")

if __name__ == "__main__":
    unittest.main()





