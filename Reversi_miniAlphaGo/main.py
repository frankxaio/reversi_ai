import copy, rich
from copy import deepcopy

from rich import print
from rich.console import Console

import random

from board import Board

import math

# 棋盤初始化
board = Board()

# 打印初始化棋盤
board.display()


class RandomPlayer:
    """
    隨機玩家, 隨機返回一個合法落子位置
    """

    def __init__(self, color):
        """
        玩家初始化
        :param color: 下棋方，'X' - 黑棋，'O' - 白棋
        """
        self.color = color

    def random_choice(self, board):
        """
        從合法落子位置中隨機選一個落子位置
        :param board: 棋盤
        :return: 隨機合法落子位置, e.g. 'A1'
        """
        # 用 list() 方法獲取所有合法落子位置坐標列表
        action_list = list(board.get_legal_actions(self.color))

        # 如果 action_list 為空，則返回 None,否則從中選取一個隨機元素，即合法的落子坐標
        if len(action_list) == 0:
            return None
        else:
            return random.choice(action_list)

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
        action = self.random_choice(board)
        return action


class HumanPlayer:
    """
    人類玩家
    """

    def __init__(self, color):
        """
        玩家初始化
        :param color: 下棋方，'X' - 黑棋，'O' - 白棋
        """
        self.color = color

    def get_move(self, board):
        """
        根據當前棋盤輸入人類合法落子位置
        :param board: 棋盤
        :return: 人類下棋落子位置
        """
        # 如果 self.color 是黑棋 "X",則 player 是 "黑棋"，否則是 "白棋"
        if self.color == "X":
            player = "黑棋"
        else:
            player = "白棋"

        # 人類玩家輸入落子位置，如果輸入 'Q', 則返回 'Q'並結束比賽。
        # 如果人類玩家輸入棋盤位置，e.g. 'A1'，
        # 首先判斷輸入是否正確，然後再判斷是否符合黑白棋規則的落子位置
        while True:
            action = input(
                "請'{}-{}'方輸入一個合法的坐標(e.g. 'D3'，若不想進行，請務必輸入'Q'結束遊戲。): ".format(player,
                                                                             self.color))

            # 如果人類玩家輸入 Q 則表示想結束比賽
            if action == "Q" or action == 'q':
                return "Q"
            elif len(action) == 1:
                action = input("請'{}-{}'方輸入一個合法的坐標(e.g. 'D3'，若不想進行，請務必輸入'Q'結束遊戲。): ".format(player,
                                                                             self.color))
            else:
                row, col = action[1].upper(), action[0].upper()

                # 檢查人類輸入是否正確
                if row in '12345678' and col in 'ABCDEFGH':
                    # 檢查人類輸入是否為符合規則的可落子位置
                    if action in board.get_legal_actions(self.color):
                        return action
                else:
                    print("你的輸入不合法，請重新輸入!")


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


from game import Game

# 人類玩家黑棋初始化
# black_player = AIPlayer("X", 60)
# black_player  = HumanPlayer("X")
black_player  = AIPlayer("X", 500)

# AI 玩家 白棋初始化
white_player = AIPlayer("O", 100)

# 遊戲初始化，第一個玩家是黑棋，第二個玩家是白棋
game = Game(black_player, white_player)

# 開始下棋
game.run()
