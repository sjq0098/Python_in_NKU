##overview    
@startuml   
|#LightBlue|数据读取|#LightBlue| -> |#LightYellow|数据预处理|#LightYellow| : 读取Sleep_health_and_lifestyle.csv文件   
|#LightYellow|数据预处理|#LightYellow| -> |#LightGreen|模型构建与训练|#LightGreen| : 划分特征和目标变量\n进行数据标准化和独热编码\n创建模型管道并训练多个模型   
|#LightGreen|模型构建与训练|#LightGreen| -> |#LightOrange|模型评估|#LightOrange| : 计算各模型的MSE、MAE、R2分数    
|#LightOrange|模型评估|#LightOrange| -> |#LightRed|模型选择|#LightRed| : 根据R2分数选择最佳模型     
|#LightRed|模型选择|#LightRed| -> |#LightBlue|新数据预测|#LightBlue| : 使用最佳模型预测新数据点的睡眠时间    
|#LightBlue|新数据预测|#LightBlue| -> |#LightViolet|结果可视化与分析|#LightViolet| : 可视化预测结果与真实值\n计算并打印预测准确率    
|#LightViolet|结果可视化与分析|#LightViolet| -> |#LightGray|结果保存|#LightGray| : 保存可视化结果图和评估指标    
@enduml    
