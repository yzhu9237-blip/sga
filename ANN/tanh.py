import torch
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

_, axes = plt.subplots(1, 2)

# 函数图像
def dm01():
    x = torch.linspace(-20, 20, 1000)
    y = torch.tanh(x)
    axes[0].plot(x, y)
    axes[0].grid()
    axes[0].set_title('Tanh 函数图像')
    # 导数图像
    x = torch.linspace(-20, 20, 1000, requires_grad=True)
    torch.tanh(x).sum().backward()
    axes[1].plot(x.detach(), x.grad)
    axes[1].grid()
    axes[1].set_title('Tanh 导数图像')
    plt.show()
def dm02():
    # 测试数据
    # （2，5） 2个样本 每个样本5个特征
    x= torch.randn(size=(2,5))
    # 实例化线性层对象
    # 参数1： 上一层的特征数一致  参数2：当前层的特征数
    l1=torch.nn.Linear(in_features=5,out_features=3)