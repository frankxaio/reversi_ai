"""蒙特卡洛樹搜索"""


import numpy as np
import copy
from config import CONFIG
from model import PolicyValueNet
from board import Board


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


# 定義葉子節點
class TreeNode(object):
    """
    mcts樹中的節點，樹的子節點字典中，鍵為動作，值為TreeNode。記錄當前節點選擇的動作，以及選擇該動作後會跳轉到的下一個子節點。
    每個節點跟蹤其自身的Q，先驗概率P及其訪問次數調整的u
    """

    def __init__(self, parent, prior_p):
        """
        :param parent: 當前節點的父節點
        :param prior_p:  當前節點被選擇的先驗概率，也就是網路的預測機率輸出
        """
        self._parent = parent
        self._children = {} # 從動作到TreeNode的映射
        self._n_visits = 0  # 當前當前節點的訪問次數
        self._Q = 0         # 當前節點對應動作的平均動作價值
        self._u = 0         # 當前節點的置信上限         # PUCT算法進行節點選擇
        self._P = prior_p

    def expand(self, action_priors):    # 這里把不合法的動作概率全部設置為0
        """通過創建新子節點來展開樹"""
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] =  TreeNode(self, prob) # 當前節點傳到 parent

    def select(self, c_puct):
        """
        在子節點中選擇能夠提供最大的Q+U的節點，也就是控制 Q+U 的值
        return: (action, next_node)的二元組
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def get_value(self, c_puct):
        """
        計算並返回此節點的值，它是節點評估Q和此節點的先驗的組合
        c_puct: 控制相對影響（0， inf）
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def update(self, leaf_value):
        """
        從葉節點評估中更新節點值，也就是 backpropagation 更新值
        leaf_value: 這個子節點的評估值來自當前玩家的視角
        """
        # 統計訪問次數
        self._n_visits += 1
        # 更新Q值，取決於所有訪問次數的平均樹，使用增量式更新方式
        # 葉節點是正值，就增加 Q 值，否則減少 Q 值
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

    # 使用遞歸的方法對所有節點（當前節點對應的支線）進行一次更新，只對父節點進行更新，不對旁系節點進行更新
    def update_recursive(self, leaf_value):
        """就像調用update()一樣，但是對所有直系節點進行更新"""
        # 如果它不是根節點，則應首先更新此節點的父節點
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def is_leaf(self):
        """檢查是否是葉節點，即沒有被擴展的節點"""
        return self._children == {}

    def is_root(self):
        return self._parent is None


# 蒙特卡洛搜索樹
class MCTS(object):

    def __init__(self, policy_value_fn, c_puct=5, n_playout=2000):
        """
        policy_value_fn: 接收board的盤面狀態，返回落子概率和盤面評估得分
        c_puct: PUCT算法中的探索參數
        n_playout: 每次走子模擬的次數
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        """
        進行一次搜索，根據葉節點的評估值進行反向更新樹節點的參數
        注意：state已就地修改，因此必須提供副本
        """
        node = self._root
        while True:
            if node.is_leaf():
                break
            # 貪心算法選擇下一步行動
            action, node = node.select(self._c_puct)
            state._move(action)

        # 使用網絡評估葉子節點，網絡輸出（動作，概率）元組p的列表以及當前玩家視角的得分[-1, 1]
        action_probs, leaf_value = self._policy(state)
        # 查看遊戲是否結束
        if state.game_over():
            # 對於結束狀態，將葉子節點的值換成1或-1
            winner, diff = state.get_winner()
            if winner == 2:    # Tie
                leaf_value = 0.0
            else:
                leaf_value = (
                    1.0 if winner == state.current_player() else -1.0
                )
        else:
            node.expand(action_probs)
        # 在本次遍歷中更新節點的值和訪問次數
        # 必須添加符號，因為兩個玩家共用一個搜索樹
        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp=1e-3):
        """
        按順序運行所有搜索並返回可用的動作及其相應的概率
        state:當前遊戲的狀態
        temp:介於（0， 1]之間的溫度參數
        """
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        # 跟據根節點處的訪問計數來計算移動概率
        act_visits= [(act, node._n_visits)
                     for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
        return acts, act_probs

    def update_with_move(self, last_move):
        """
        在當前的樹上向前一步，保持我們已經直到的關於子樹的一切
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return 'MCTS'


# 基於MCTS的AI玩家
class MCTSPlayer(object):

    def __init__(self, policy_value_function, c_puct=5, n_playout=2000, is_selfplay=0):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay
        self.agent = "AI"

    def set_player_ind(self, p):
        self.player = p

    # 重置搜索樹
    def reset_player(self):
        self.mcts.update_with_move(-1)

    def __str__(self):
        return 'MCTS {}'.format(self.player)

    # 得到行動
    def get_action(self, board, temp=1e-3, return_prob=0):
        # 像alphaGo_Zero論文一樣使用MCTS算法返回的pi向量
        move_probs = np.zeros(64)

        acts, probs = self.mcts.get_move_probs(board, temp)
        move_probs[list(acts)] = probs
        if self._is_selfplay:
            # 添加Dirichlet Noise進行探索（自我對弈需要）
            move = np.random.choice(
                acts,
                p=0.75*probs + 0.25*np.random.dirichlet(CONFIG['dirichlet'] * np.ones(len(probs)))
            )
            # 更新根節點並重用搜索樹
            self.mcts.update_with_move(move)
        else:
            # 使用默認的temp=1e-3，它幾乎相當於選擇具有最高概率的移動
            move = np.random.choice(acts, p=probs)
            # 重置根節點
            self.mcts.update_with_move(-1)
        if return_prob:
            return move, move_probs
        else:
            return move

if __name__ == '__main__':
    board = Board()  # 棋盤初始化
    # 印出棋盤
    # board.display()
    policy_value_net=PolicyValueNet()
    mcts=MCTSPlayer(policy_value_net.policy_value_fn(board,'black'))
    mcts.get_action(board)