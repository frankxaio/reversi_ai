from board import Board
import math
import random
import copy
from copy import deepcopy
from rich import print
from rich.console import Console
import time


class Node:
    def __init__(self, state, color='X', parent=None, action=None):
        self.state = state
        self.color = color
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.reward = 0.0

    def AddChild(self, state_c, color, action):
        child = Node(state_c, color, self, action)
        self.children.append(child)
        return child


c = pow(2, 0.5) / 2


def nextColor(color):
    if color == 'X':
        new_color = 'O'
    else:
        new_color = 'X'
    return new_color


def eqBoard(board1: Board, board2: Board):
    for i in range(8):
        if board1.__getitem__(i) != board2.__getitem__(i):
            return False
    return True


# priority = [[1, 5, 3, 3, 3, 3, 5, 1],
#             [5, 5, 4, 4, 4, 4, 5, 5],
#             [3, 4, 2, 2, 2, 2, 4, 3],
#             [3, 4, 2, 2, 2, 2, 4, 3],
#             [3, 4, 2, 2, 2, 2, 4, 3],
#             [3, 4, 2, 2, 2, 2, 4, 3],
#             [5, 5, 4, 4, 4, 4, 5, 5],
#             [1, 5, 3, 3, 3, 3, 5, 1]]

priority = [[5, 1, 3, 3, 3, 3, 1, 5],
            [1, 1, 2, 2, 2, 2, 1, 1],
            [3, 2, 4, 4, 4, 4, 2, 3],
            [3, 2, 4, 4, 4, 4, 2, 3],
            [3, 2, 4, 4, 4, 4, 2, 3],
            [3, 2, 4, 4, 4, 4, 2, 3],
            [1, 1, 2, 2, 2, 2, 1, 1],
            [5, 1, 3, 3, 3, 3, 1, 5]]


class AIPlayer:
    """
    AI 玩家
    """

    def __init__(self, color, max_times=8000):
        """
        玩家初始化
        :param color: 下棋方，'X' - 黑棋，'O' - 白棋
        """

        self.color = color
        self.max_times = max_times
        board_now = Board()
        self.node = Node(state=board_now, color=self.color)

    def get_move(self, board):
        """
        根據當前棋盤狀態獲取最佳落子位置
        :param board: 棋盤
        :return: action 最佳落子位置, e.g. 'A1'
        """
        if self.color == 'X':
            player_name = '黑棋'
        else:
            player_name = '白棋'
        print("請等一會，對方 {}-{} 正在思考中...".format(player_name, self.color))

        # -----------------請實現你的算法代碼--------------------------------------
        root = None
        for node in self.node.children:
            if eqBoard(node.state, board):
                root = node
                break
        if root is None:
            board_now = deepcopy(board)
            root = Node(state=board_now, color=self.color)
        self.node = root
        action = self.UCTSearch(max_times=self.max_times, node=root)
        # ------------------------------------------------------------------------
        console = Console()
        print("===========================")
        console.print(f"AI action:{action}", style = 'italic magenta')
        print("===========================")
        return action

    def UCTSearch(self, max_times, node):
        """
        通過蒙特卡洛搜索返回此時的最佳動作
        :param max_times: 允許的最大搜索次數
        :param node: 蒙特卡洛樹上當前訪問的節點,表示當前狀態
        :return action: 玩家MAX行動下最優動作a*
        """
        time_s = time.time()
        for i in range(max_times):
            leave = self.SelectPolicy(node)
            self.BackPropagate(leave)
            time_n = time.time()
            if time_n-time_s > 55:
                break
        m = -1
        best_action = None
        for a in node.children:
            if a.reward > m:
                m = a.reward
                best_action = a.action
                self.node = a
        return best_action

    def SelectPolicy(self, node):
        """
        選擇將要被拓展的節點
        :param node: 蒙特卡洛樹上當前訪問的節點
        :return node: 將要被拓展的節點
        """
        while len(list(node.state.get_legal_actions(node.color))):
            node.visits += 1
            if len(node.children) < len(list(node.state.get_legal_actions(node.color))):
                temp = self.Expand(node)
                return temp
            else:
                stack = []
                max_node = None
                max_value = float(-1)
                for i in range(len(node.children)):
                    val = node.children[i].reward / node.children[i].visits + c * pow(
                        2 * math.log(node.visits) / node.children[i].visits, 0.5)
                    if val > max_value:
                        max_value = val
                        max_node = node.children[i]
                node = max_node
        return node

    def Expand(self, node):
        """
        完成節點的拓展
        :param node: 待拓展的節點
        :return node: 該拓展對應的隨機葉子節點
        """
        actions = list(node.state.get_legal_actions(node.color))
        tried = [temp.action for temp in node.children]
        to_expand = None
        val = 6
        for a in actions:
            if a in tried:
                continue
            row, col = Board.board_num(node.state, a)
            if priority[row][col] < val:
                val = priority[row][col]
                to_expand = a
        new_state = copy.deepcopy(node.state)
        new_state._move(to_expand, node.color)

        if node.color == 'X':
            new_color = 'O'
        else:
            new_color = 'X'
        return self.SimulatePolicy(node.AddChild(state_c=new_state, action=to_expand, color=new_color))

    def SimulatePolicy(self, node):
        """
        模擬當前狀態終局的結果
        :param node: 節點 表示當前狀態
        :return state: 模擬棋局得到的終局節點
        """
        board = copy.deepcopy(node.state)
        actions = list(board.get_legal_actions(node.color))
        if len(actions):
            optimal = None
            val = 6
            for a in actions:
                row, col = Board.board_num(node.state, a)
                if priority[row][col] < val:
                    val = priority[row][col]
                    optimal = a
            new_state = copy.deepcopy(node.state)
            new_state._move(optimal, node.color)
            return self.SimulatePolicy(node.AddChild(state_c=new_state, action=optimal, color=nextColor(node.color)))
        else:
            if len(list(board.get_legal_actions(nextColor(node.color)))):
                node.AddChild(state_c=node.state, action=None, color=nextColor(node.color))
                return self.SimulatePolicy(node.children[0])
            else:
                return node

    def BackPropagate(self, node):
        """
        反向傳播
        :param node: 反向傳播更新的起始節點
        :return: NONE
        """
        # res:  0-黑棋贏, 1-白旗贏, 2-表示平局，黑棋個數和白旗個數相等
        res = node.state.get_winner
        while node:
            node.visits += 1
            if res == 2:
                if node.color == self.color:
                    node.reward += +0.5
                else:
                    node.reward += -0.5
            elif res == 0:
                if self.color == 'X':
                    if node.color == self.color:
                        node.reward += +1
                    else:
                        node.reward += -1
            elif res == 1:
                if self.color == 'O':
                    if node.color == self.color:
                        node.reward += +1
                    else:
                        node.reward += -1
            node = node.parent
