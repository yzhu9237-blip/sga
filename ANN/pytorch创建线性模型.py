import torch
from torch.utils.data import TensorDataset  # 构造数据集对象
from torch.utils.data import DataLoader  # 数据加载器
from torch import nn  # nn模块中有平方损失函数和假设函数
from torch import optim  # optim模块中有优化器函数
from sklearn.datasets import make_regression  # 创建线性回归模型数据集
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 构造数据集
def create_dataset():
    x, y, coef = make_regression(n_samples=100,
                                 n_features=1,
                                 noise=10,
                                 coef=True,
                                 bias=14.5,
                                 random_state=0)

    # 将构建数据转换为张量类型
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    return x, y, coef


# 训练模型
def train():
    # 构造数据集
    x, y, coef = create_dataset()

    # 构造数据集对象
    dataset = TensorDataset(x, y)
    print('dataset--->', dataset)
    print('len(dataset)--->', len(dataset))
    # 获取数据集x和y, 返回元组类型
    print('dataset.tensors--->', dataset.tensors)
    # 获取第1条样本
    print('dataset[0]--->', dataset[0])

    # 构造数据加载器
    # dataset=:数据集对象
    # batch_size=:批量训练样本数据
    # shuffle=:样本数据是否进行乱序
    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True)
    print('dataloader--->', dataloader)
    print('len(dataloader)--->', len(dataloader))

    # 构造模型
    # in_features指的是输入的二维张量的大小，即输入的[batch_size, size]中的size
    # out_features指的是输出的二维张量的大小，即输出的[batch_size，size]中的size
    model = nn.Linear(in_features=1, out_features=1)

    # 构造平方损失函数
    criterion = nn.MSELoss()

    # 构造优化函数
    # params=model.parameters():训练的参数,w和b
    # lr=1e-2:学习率, 1e-2为10的负二次方
    print("w和b-->", list(model.parameters()))
    print("w-->", model.weight)
    print("b-->", model.bias)
    optimizer = optim.SGD(params=model.parameters(), lr=1e-2)

    # 初始化训练次数
    epochs = 100
    # 损失的变化
    epoch_loss = []
    # 总损失
    total_loss = 0.0
    # 总批次数
    iter_num = 0.0
    for _ in range(epochs):
        for train_x, train_y in dataloader:
            print('train_x--->', train_x.shape, train_x)
            print('train_y--->', train_y.shape, train_y)
            # 将一个batch的训练数据送入模型
            y_pred = model(train_x)
            # 计算损失值,均方误差,当前批次所有样本的平均误差
            loss = criterion(y_pred, train_y.reshape(-1, 1))
            total_loss += loss.item()
            # 总的批次数
            iter_num += 1
            # 梯度清零
            optimizer.zero_grad()
            # 自动微分(反向传播)
            loss.backward()
            # 更新参数
            optimizer.step()
        # 计算所有batch的平均误差作为当前epoch的误差
        epoch_loss.append(total_loss / iter_num)

    # 打印回归模型的w
    print(model.weight)
    # 打印回归模型的b
    print(model.bias)

    # 绘制损失变化曲线
    plt.plot(range(epochs), epoch_loss)
    plt.title('损失变化曲线')
    plt.grid()
    plt.show()

    # 绘制拟合直线
    plt.scatter(x, y)
    x = torch.linspace(x.min(), x.max(), 1000)
    # 根据训练出来的w和b获取预测的y值
    y1 = torch.tensor([v * model.weight + model.bias for v in x])
    # 根据真实的w和b获取真实的y值
    y2 = torch.tensor([v * coef + 14.5 for v in x])
    plt.plot(x, y1, label='训练')
    plt.plot(x, y2, label='真实')
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    train()