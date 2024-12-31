from data_manage import DataManage
class DatasetManage:
    datasets = [ #数据集信息，每个数据集有唯一的名称
        dict(name='BMI', obj=DataManage(data_list=[],iteminfo_list=[])),
        dict(name='Diabetes', obj=DataManage(data_list=[],iteminfo_list=[])),
        dict(name='CKD', obj=DataManage(data_list=[],iteminfo_list=[]))
    ]
    @classmethod
    def run(cls): #开始运行程序
        while True:
            if cls.switch_dataset()==False: #设置当前使用的数据集
                break
            cls.dataset.manage_data() #进行当前数据集的数据管理操作
        
    @classmethod
    def switch_dataset(cls): #切换数据集
        for idx in range(len(cls.datasets)): #依次每一个数据集的编号和名称
            print('%d %s'%(idx+1, cls.datasets[idx]['name']))
        dataset_no = int(input('请输入数据集编号（输入0结束程序）：'))
        if dataset_no == 0:
            return False
        if dataset_no<1 or dataset_no>len(cls.datasets):
            print('该数据集不存在！')
        cls.dataset = cls.datasets[dataset_no-1]['obj'] #设置当前使用的数据集
        return True
