import matplotlib.pyplot as plt

def plot_models_comparison():
    plt.style.use('seaborn-whitegrid')
    
    # 原有模型数据(根据前面估计的数据，可根据实际需要修改)
    models = {
        "Informer (2021)":    (2e7,   0.385),
        "Autoformer (2021)":  (3e7,   0.24),
        "FEDformer (2022)":   (1.5e7,   0.26),
        "FiLM (2022)":        (1.5e7,   0.23),
        "PatchTST (2023)":    (8e6,   0.195),
        "DLinear (2023)":     (7.5e4,   0.20),
        "FITS (2024)":        (6.5e5,  0.23),
        "SparseTSF (2024)":   (1000,  0.209),
        "SparseGraphTSF (Ours)": (1000, 0.20),
    }


    # 标记和颜色对应关系，可根据需要调整
    model_styles = {
        "Informer (2021)":    {'color': 'red',     'marker': 'o'},
        "Autoformer (2021)":  {'color': 'yellow',  'marker': 's'},
        "FEDformer (2022)":   {'color': 'purple',  'marker': 'D'},
        "FiLM (2022)":        {'color': 'green',   'marker': '*'},
        "PatchTST (2023)":    {'color': 'blue',    'marker': '^'},
        "DLinear (2023)":     {'color': 'cyan',    'marker': 'v'},
        "FITS (2024)":        {'color': 'teal',    'marker': 'P'},
        "SparseTSF (2024)":   {'color': 'orange',  'marker': 'v'},
        "SparseGraphTSF (Ours)": {'color': 'magenta', 'marker': 'X'}
    }

    names = list(models.keys())
    parameters = [models[name][0] for name in names]
    mse_values = [models[name][1] for name in names]

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制点
    for name in names:
        param, mse = models[name]
        style = model_styles.get(name, {'color': 'black', 'marker': 'o'})
        # 如果是 Ours，则单独标注
        if "Ours" in name:
            ax.scatter(param, mse, color=style['color'], marker=style['marker'], s=100, alpha=0.9, label=name)
        else:
            ax.scatter(param, mse, color=style['color'], marker=style['marker'], s=100, alpha=0.9)

    # 首先移除默认图例
    ax.legend().remove()

    # 重新设置图例
    # 我们添加两类图例：已有模型和 Ours 模型。
    # 若想详细区分各模型，可直接使用前面生成的图例项，但这里演示较简洁方式。
    ax.scatter([], [], color='gray', marker='o', label='Existing Models')
    ax.scatter([], [], color='orange', marker='v', label='SparseTSF (2024)')
    ax.scatter([], [], color='magenta', marker='X', label='SparseGraphTSF (Ours)')
    ax.legend(fontsize=11)

    # 添加注释
    for i, name in enumerate(names):
        param = parameters[i]
        mse = mse_values[i]
        x_offset = 1.0
        y_offset = 0.002
        if "Ours" in name:
            x_offset = 1.0
            y_offset = -0.008
        ax.annotate(name,
                    (param, mse),
                    xytext=(param * 1.2, mse + y_offset),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.0, shrinkA=5, shrinkB=5),
                    fontsize=10)

    ax.set_xscale('log')
    ax.set_xlabel("Parameters", fontsize=12)
    ax.set_ylabel("Mean Squared Error (MSE)", fontsize=12)
    ax.set_title("Performance and Parameter Count Comparison", fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_models_comparison()

