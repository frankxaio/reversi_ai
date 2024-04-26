import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from config import CONFIG
from torch.cuda.amp import autocast


# 搭建殘差塊
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
    象棋棋盤是高 10 寬 9 的盤面，所以輸入是 9 * 10 * 9
    搭建骨幹網絡，輸入：N, 9, 10, 9 --> N, C, H, W
    N: 代表 batch_size，C 代表通道數，H 代表高度，W 代表寬度
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
        # 殘差塊抽取特征
        self.res_blocks = nn.ModuleList([ResBlock(num_filters=num_channels) for _ in range(num_res_blocks)])
        # 策略頭
        # 代表落子的機率
        self.policy_conv = nn.Conv2d(in_channels=num_channels, out_channels=16, kernel_size=(1, 1), stride=(1, 1))
        self.policy_bn = nn.BatchNorm2d(16)
        self.policy_act = nn.ReLU()
        self.policy_fc = nn.Linear(16 * 8 * 8, 8*8)
        # 價值頭
        # 代表獲勝的機率
        self.value_conv = nn.Conv2d(in_channels=num_channels, out_channels=8, kernel_size=(1, 1), stride=(1, 1))
        self.value_bn = nn.BatchNorm2d(8)
        self.value_act1 = nn.ReLU()
        self.value_fc1 = nn.Linear(8 * 8 * 8, 256)
        self.value_act2 = nn.ReLU()
        self.value_fc2 = nn.Linear(256, 1)

    # 定義前向傳播
    def forward(self, x):
        # 公共頭
        x = self.conv_block(x)
        x = self.conv_block_bn(x)
        x = self.conv_block_act(x)
        for layer in self.res_blocks:
            x = layer(x)
        # 策略頭
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = self.policy_act(policy)
        policy = torch.reshape(policy, [-1, 16 * 8 * 8])
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy)
        # 價值頭
        value = self.value_conv(x)
        value = self.value_bn(value)
        value = self.value_act1(value)
        value = torch.reshape(value, [-1, 8 * 8 * 8])
        value = self.value_fc1(value)
        value = self.value_act1(value)
        value = self.value_fc2(value)
        value = F.tanh(value)

        return policy, value

# 策略值網絡，用來進行模型的訓練
class PolicyValueNet:

    def __init__(self, model_file=None, use_gpu=True, device = 'cuda'):
        self.use_gpu = use_gpu
        self.l2_const = 2e-3    # l2 正則化
        self.device = device
        self.policy_value_net = Net().to(self.device)
        self.optimizer = torch.optim.Adam(params=self.policy_value_net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=self.l2_const)
        if model_file:
            self.policy_value_net.load_state_dict(torch.load(model_file))  # 加載模型參數

    # 輸入一個批次的狀態，輸出一個批次的動作概率和狀態價值
    def policy_value(self, state_batch):
        self.policy_value_net.eval()
        state_batch = torch.tensor(state_batch).to(self.device)
        log_act_probs, value = self.policy_value_net(state_batch)
        log_act_probs, value = log_act_probs.cpu(), value.cpu()
        act_probs = np.exp(log_act_probs.detach().numpy())
        return act_probs, value.detach().numpy()

    # 輸入棋盤，返回每個合法動作的（動作，概率）元組列表，以及棋盤狀態的分數
    def policy_value_fn(self, board, color):
        self.policy_value_net.eval()
        # 獲取合法動作列表
        board.display
        legal_positions = list(board.get_legal_actions(color))
        print(legal_positions)
        # reshape 成 C*H*W
        current_state = np.ascontiguousarray(np.array(board.one_hot_encoding()).reshape(-1, 3, 4, 16)).astype('float16')
        current_state = torch.as_tensor(current_state).to(self.device)
        # 使用神經網絡進行預測
        with autocast(): #半精度fp16
            log_act_probs, value = self.policy_value_net(current_state)
        log_act_probs, value = log_act_probs.cpu() , value.cpu()
        act_probs = np.exp(log_act_probs.numpy().flatten()) if CONFIG['use_frame'] == 'paddle' else np.exp(log_act_probs.detach().numpy().astype('float16').flatten())
        # 只取出合法動作
        act_probs = zip(legal_positions, act_probs[legal_positions])
        # 返回動作概率，以及狀態價值
        return act_probs, value.detach().numpy()

    # 保存模型
    def save_model(self, model_file):
        torch.save(self.policy_value_net.state_dict(), model_file)

    # 執行一步訓練
    def train_step(self, state_batch, mcts_probs, winner_batch, lr=0.002):
        self.policy_value_net.train()
        # 包裝變量
        state_batch = torch.tensor(state_batch).to(self.device)
        mcts_probs = torch.tensor(mcts_probs).to(self.device)
        winner_batch = torch.tensor(winner_batch).to(self.device)
        # 清零梯度
        self.optimizer.zero_grad()
        # 設置學習率
        for params in self.optimizer.param_groups:
            # 遍歷Optimizer中的每一組參數，將該組參數的學習率 * 0.9
            params['lr'] = lr
        # 前向運算
        log_act_probs, value = self.policy_value_net(state_batch)
        value = torch.reshape(value, shape=[-1])
        # 價值損失
        value_loss = F.mse_loss(input=value, target=winner_batch)
        # 策略損失
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, dim=1))  # 希望兩個向量方向越一致越好
        # 總的損失，注意l2懲罰已經包含在優化器內部
        loss = value_loss + policy_loss
        # 反向傳播及優化
        loss.backward()
        self.optimizer.step()
        # 計算策略的熵，僅用於評估模型
        with torch.no_grad():
            entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, dim=1)
            )
        return loss.detach().cpu().numpy(), entropy.detach().cpu().numpy()


if __name__ == '__main__':
    net = Net().to('cuda')
    test_data = torch.ones([8, 3, 8, 8]).to('cuda')
    x_act, x_val = net(test_data)
    print(f"action: {x_act.shape}")  # 8, 64
    print(f"value: {x_val.shape}")  # 8, 1

# if __name__ == '__main__':