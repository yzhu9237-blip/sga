import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        # 调用父类方法
        super().__init__()
    # 定义结构
    l1=nn.Linear(3,3)
    # dropout 层
    # p 是概率 每个神经元死亡的概率
    dropout1=nn.Dropout(p=0.1)

    def forward(self,x):
        # 线性计算
        x=self.l1(x)
        # 激活计算
        x=torch.tanh(x)
        # dropout 计算
        x=self.dropout1(x)
        return x
if __name__ == '__main__':
    model1=Model()
    x=torch.randn(size=(5,3))
    print(model1.forward(x))
