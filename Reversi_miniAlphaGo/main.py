import copy
from copy import deepcopy

from rich import print
from rich.console import Console
import MCTS_Player

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


from game import Game

AIPlayer1 = MCTS_Player.AIPlayer("X", 2000)
AIPlayer2 = MCTS_Player.AIPlayer("X", 1000)
# 人類玩家黑棋初始化
# black_player  = HumanPlayer("X")
black_player  = AIPlayer2

# AI 玩家 白棋初始化
white_player = AIPlayer1

# 遊戲初始化，第一個玩家是黑棋，第二個玩家是白棋
game = Game(black_player, white_player)

# 開始下棋
game.run()
