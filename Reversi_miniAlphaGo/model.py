import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# from config import CONFIG
from torch.cuda.amp import autocast


# 搭建残差块
class ResBlock(nn.Module):

    def __init__(self, num_filters=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv1_bn = nn.BatchNorm2d(num_filters, )
        self.conv1_act = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv2_bn = nn.BatchNorm2d(num_filters, )
        self.conv2_act = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv1_bn(y)
        y = self.conv1_act(y)
        y = self.conv2(y)
        y = self.conv2_bn(y)
        y = x + y
        return self.conv2_act(y)


class Net(nn.Module):
    """
    象棋棋盤是高 10 寬 9 的盤面，所以输入是 9 * 10 * 9
    搭建骨干网络，输入：N, 9, 10, 9 --> N, C, H, W
    N: 代表 batch_size，C 代表通道数，H 代表高度，W 代表宽度
    C: 通道數量，使用 7 個通道以 one-hot 方式表示棋盤上的棋子, 再 +1 表示棋盤上全部的旗子，再 +1 表示上一步的落子位置，觀察對手的動向。
    翻轉棋遊戲的話使用 N, 8, 8 的輸入, 以 -1 表示對方子，0 表示空格，1 表示自己的子
    """
    def __init__(self, num_channels=256, num_res_blocks=7):
        super().__init__()
        # 全局特征
        # self.global_conv = nn.Conv2D(in_channels=9, out_channels=512, kernel_size=(10, 9))
        # self.global_bn = nn.BatchNorm2D(512)
        # 初始化特征
        self.conv_block = nn.Conv2d(in_channels=3, out_channels=num_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv_block_bn = nn.BatchNorm2d(256)
        self.conv_block_act = nn.ReLU()
        # 残差块抽取特征
        self.res_blocks = nn.ModuleList([ResBlock(num_filters=num_channels) for _ in range(num_res_blocks)])
        # 策略头
        self.policy_conv = nn.Conv2d(in_channels=num_channels, out_channels=16, kernel_size=(1, 1), stride=(1, 1))
        self.policy_bn = nn.BatchNorm2d(16)
        self.policy_act = nn.ReLU()
        self.policy_fc = nn.Linear(16 * 8 * 8, 2086)
        # 价值头
        self.value_conv = nn.Conv2d(in_channels=num_channels, out_channels=8, kernel_size=(1, 1), stride=(1, 1))
        self.value_bn = nn.BatchNorm2d(8)
        self.value_act1 = nn.ReLU()
        self.value_fc1 = nn.Linear(8 * 8 * 8, 256)
        self.value_act2 = nn.ReLU()
        self.value_fc2 = nn.Linear(256, 1)

    # 定义前向传播
    def forward(self, x):
        # 公共头
        x = self.conv_block(x)
        x = self.conv_block_bn(x)
        x = self.conv_block_act(x)
        for layer in self.res_blocks:
            x = layer(x)
        # 策略头
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = self.policy_act(policy)
        policy = torch.reshape(policy, [-1, 16 * 8 * 8])
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy)
        # 价值头
        value = self.value_conv(x)
        value = self.value_bn(value)
        value = self.value_act1(value)
        value = torch.reshape(value, [-1, 8 * 8 * 8])
        value = self.value_fc1(value)
        value = self.value_act1(value)
        value = self.value_fc2(value)
        value = F.tanh(value)

        return policy, value

# if __name__ == '__main__':