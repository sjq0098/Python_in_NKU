## overviewing
project_root/
│
├── data/                # 数据文件
│   ├── raw/             # 原始数据
│   ├── processed/       # 处理后的数据
│   └── external/        # 外部数据集（如果有）
│
├── notebooks/           # Jupyter Notebook文件
│   └── exploration.ipynb  # 数据探索和可视化
│
├── src/                # 源代码
│   ├── __init__.py      # 使该文件夹成为Python包
│   ├── data_preprocessing.py  # 数据预处理脚本
│   ├── model.py         # 模型定义和训练脚本
│   ├── train.py         # 训练流程脚本
│   ├── evaluate.py      # 评估脚本
│   └── utils.py         # 辅助函数（如数据可视化等）
│
├── results/            # 结果文件
│   ├── figures/         # 可视化结果图
│   └── metrics.csv      # 评估指标
│
├── requirements.txt     # 项目依赖
└── README.md            # 项目说明文档

## getting start

conda create -n ex python=3.11
conda activate ex
pip install -r requirements.txt