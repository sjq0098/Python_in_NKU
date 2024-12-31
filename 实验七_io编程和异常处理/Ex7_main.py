from Ex7_function import create_signal_handler,register_user, read_user_data, write_user_data, delete_user_data, modify_user_data, login
import signal
import sys

def main():
    
    file_path = "./实验七_io编程和异常处理/user_data.txt"
    userdata = {}
    read_user_data(userdata, file_path)
    signal.signal(signal.SIGINT, create_signal_handler(userdata, file_path))  # 处理Ctrl+C退出信号
    while True:
        print("请输入需要的操作:")
        print("1. 注册")
        print("2. 登录")
        print("3. 修改密码")
        print("4. 注销")
        print("5. 退出")
        
        choice = input()
        
        if choice == "1":
            register_user(userdata, file_path)
            read_user_data(userdata, file_path)  # 确保注册后数据加载到内存
        elif choice == "2":
            login(userdata)
        elif choice == "3":
            username = input("请输入用户名：")
            new_password = input("请输入新的密码：")
            modify_user_data(username, new_password, userdata, file_path)
        elif choice == "4":
            username = input("请输入要注销的用户名：")
            delete_user_data(username, userdata, file_path)
        elif choice == "5":
            write_user_data(userdata, file_path)
            print("退出程序")
            break
        else:
            print("输入错误，请重新输入！")

if __name__ == "__main__":
    main()



