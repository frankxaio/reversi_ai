# !/usr/bin/Anaconda3/python
# -*- coding: utf-8 -*-

from func_timeout import func_timeout, FunctionTimedOut
import datetime
from board import Board
from copy import deepcopy
from rich import print
import time
import numpy as np


class Game(object):
    def __init__(self, black_player, white_player):
        self.board = Board()  # 棋盤
        # 定義棋盤上當前下棋棋手，先默認是 None
        self.current_player = None
        self.black_player = black_player  # 黑棋一方
        self.white_player = white_player  # 白棋一方
        self.black_player.color = "X"
        self.white_player.color = "O"

    def switch_player(self, black_player, white_player):
        """
        遊戲過程中切換玩家
        :param black_player: 黑棋
        :param white_player: 白棋
        :return: 當前玩家
        """
        # 如果當前玩家是 None 或者 白棋一方 white_player，則返回 黑棋一方 black_player;
        if self.current_player is None:
            return black_player
        else:
            # 如果當前玩家是黑棋一方 black_player 則返回 白棋一方 white_player
            if self.current_player == self.black_player:
                return white_player
            else:
                return black_player

    def game_over(self):
        """
        判斷遊戲是否結束
        :return: True/False 遊戲結束/遊戲沒有結束
        """

        # 根據當前棋盤，判斷棋局是否終止
        # 如果當前選手沒有合法下棋的位子，則切換選手；如果另外一個選手也沒有合法的下棋位置，則比賽停止。
        b_list = list(self.board.get_legal_actions('X'))
        w_list = list(self.board.get_legal_actions('O'))
        is_over = len(b_list) == 0 and len(w_list) == 0  # 返回值 True/False

        return is_over

    def print_winner(self, winner):
        """
        打印贏家
        :param winner: [0,1,2] 分別代表黑棋獲勝、白棋獲勝、平局3種可能。
        :return:
        """
        print(['黑棋獲勝!', '白棋獲勝!', '平局'][winner])

    def run(self):
        """
        運行遊戲，可以讓 AI 自動下棋，也可以讓人類下棋
        :return:
        """
        # 定義統計雙方下棋時間
        total_time = {"X": 0, "O": 0}
        # 定義雙方每一步下棋時間
        step_time = {"X": 0, "O": 0}
        # 初始化勝負結果和棋子差
        winner = None
        diff = -1

        # 遊戲開始
        print('\n=====開始遊戲!=====\n')
        # 棋盤初始化
        self.board.display(step_time, total_time)
        while True:
            # 切換當前玩家,如果當前玩家是 None 或者白棋 white_player，則返回黑棋 black_player;
            #  否則返回 white_player。
            self.current_player = self.switch_player(self.black_player, self.white_player)
            start_time = datetime.datetime.now()
            # 當前玩家對棋盤進行思考後，得到落子位置
            # 判斷當前下棋方
            color = "X" if self.current_player == self.black_player else "O"
            # 獲取當前下棋方合法落子位置
            legal_actions = list(self.board.get_legal_actions(color))
            # print(f"legal_actions:{legal_actions}")
            print("%s合法落子坐標列表："%color,legal_actions)
            if len(legal_actions) == 0:
                # 判斷遊戲是否結束
                if self.game_over():
                    # 遊戲結束，雙方都沒有合法位置
                    winner, diff = self.board.get_winner()  # 得到贏家 0,1,2
                    break
                else:
                    # 另一方有合法位置,切換下棋方
                    continue

            board = deepcopy(self.board._board)

            # legal_actions 不等於 0 則表示當前下棋方有合法落子位置
            try:
                for i in range(0, 3):
                    # 獲取落子位置
                    action = func_timeout(6000, self.current_player.get_move,
                                          kwargs={'board': self.board})
                    # 如果 action 是 Q 則說明人類想結束比賽
                    if action == "Q":
                        # 說明人類想結束遊戲，即根據棋子個數定輸贏。
                        break
                    if action not in legal_actions:
                        # 判斷當前下棋方落子是否符合合法落子,如果不合法,則需要對方重新輸入
                        print("你落子不符合規則,請重新落子！")
                        continue
                    else:
                        # 落子合法則直接 break
                        break
                else:
                    pass
                    # 落子3次不合法，結束遊戲！
                    # winner, diff = self.force_loss(is_legal=True)
                    # break
            except FunctionTimedOut:
                # 落子超時，結束遊戲
                pass
                # winner, diff = self.force_loss(is_timeout=True)
                # break

            # 結束時間
            end_time = datetime.datetime.now()
            if board != self.board._board:
                # 修改棋盤，結束遊戲！
                winner, diff = self.force_loss(is_board=True)
                break
            if action == "Q":
                # 說明人類想結束遊戲，即根據棋子個數定輸贏。
                winner, diff = self.board.get_winner()  # 得到贏家 0,1,2
                break

            if action is None:
                continue
            else:
                # 統計一步所用的時間
                es_time = (end_time - start_time).seconds
                if es_time > 600:
                    # 該步超過60秒則結束比賽。
                    print('\n{} 思考超過 60s'.format(self.current_player))
                    # winner, diff = self.force_loss(is_timeout=True)
                    # break

                # 當前玩家顏色，更新棋局
                self.board._move(action, color)
                # 統計每種棋子下棋所用總時間
                if self.current_player == self.black_player:
                    # 當前選手是黑棋一方
                    step_time["X"] = es_time
                    total_time["X"] += es_time
                else:
                    step_time["O"] = es_time
                    total_time["O"] += es_time
                # 顯示當前棋盤
                self.board.display(step_time, total_time)

                # 判斷遊戲是否結束
                if self.game_over():
                    # 遊戲結束
                    winner, diff = self.board.get_winner()  # 得到贏家 0,1,2
                    break

        print('\n=====遊戲結束!=====\n')
        self.board.display(step_time, total_time)
        self.print_winner(winner)

        # 返回'black_win','white_win','draw',棋子數差
        if winner is not None and diff > -1:
            result = {0: 'black_win', 1: 'white_win', 2: 'draw'}[winner]

            # return result,diff

     # TODO 訓練玩遊戲
    def start_self_play(self, is_shown=False, temp=1e-3):
        # 初始化勝負結果和棋子差
        winner = None
        diff = -1
        # 遊戲開始
        print('\n=====開始遊戲!=====\n')
        # 棋盤初始化
        self.board.display()
        states, mcts_probs, current_players = [], [], []
        # 开始自我对弈
        _count = 0
        while True:
            _count += 1
            if _count % 20 == 0:
                start_time = time.time()
                move, move_probs = player.get_action(self.board,
                                                     temp=temp,
                                                     return_prob=1)
                print('走一步要花: ', time.time() - start_time)
            else:
                move, move_probs = player.get_action(self.board,
                                                     temp=temp,
                                                     return_prob=1)
            # 保存自我对弈的数据
            states.append(self.board.one_hot_encoding())
            mcts_probs.append(move_probs)
            self.current_player = self.switch_player(self.black_player, self.white_player)
            # 判斷當前下棋方
            color = "X" if self.current_player == self.black_player else "O"
            # 獲取當前下棋方合法落子位置
            legal_actions = list(self.board.get_legal_actions(color))
            current_players.append(self.current_player)
            # 执行一步落子
            action = func_timeout(6000, self.current_player.get_move,
                                  kwargs={'board': self.board})
            self.board._move(action, color)
            if self.game_over():
                winner, diff = self.board.get_winner()  # 得到贏家 0,1,2
                # 从每一个状态state对应的玩家的视角保存胜负信息
                # winner == 0 代表黑子獲勝，np.zeros 代表這裡是訓練黑子方的 AI。
                winner_x = np.zeros(len(current_players))
                if winner != 2: # -1 表示有沒有贏家代表平局的狀態，只要不是平局就執行以下動作，這裡將 -1 改成 2，來表示黑白棋的平局狀況
                    winner_x[np.array(current_players) == winner] = 1.0
                    winner_x[np.array(current_players) != winner] = -1.0
                # 重置蒙特卡洛根节点
                player.reset_player()
                if is_shown:
                    if winner != 2:
                        print("Game end. Winner is:", winner)
                    else:
                        print('Game end. Tie')

                return winner, zip(states, mcts_probs, winner_x)



if __name__ == '__main__':
    # from Human_player import HumanPlayer
    # from Random_player import RandomPlayer
    # from AIPlayer import AIPlayer

    game = Game()
    game.start_self_play()
