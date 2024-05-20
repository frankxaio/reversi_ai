import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import itertools as it
from collections import defaultdict, deque
from tqdm import trange, tqdm
from colorama import Fore, Style

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

class Board(object):
    DIRECTIONS = list(it.product(*[(-1, 0, 1)] * 2))
    DIRECTIONS.remove((0, 0))
    POW2 = 2 ** torch.tensor(range(64), dtype=torch.long)

    def __init__(self, n=8, board=None):
        self.board_disp_ij = -1
        assert n == 8
        if board is None:
            self.n = n
            self.board = torch.zeros((n, n), dtype=torch.int8)
            # 人類為先手(X), AI 為後手(O)
            # self.board[n // 2 - 0, n // 2 - 0] = -1
            # self.board[n // 2 - 0, n // 2 - 1] = 1
            # self.board[n // 2 - 1, n // 2 - 0] = 1
            # self.board[n // 2 - 1, n // 2 - 1] = -1


            # 人類為後手(O)，AI 為先手(X)
            self.board[n // 2 - 0, n // 2 - 0] = 1
            self.board[n // 2 - 0, n // 2 - 1] = -1
            self.board[n // 2 - 1, n // 2 - 0] = -1
            self.board[n // 2 - 1, n // 2 - 1] = 1
        else:
            self.board = board
            self.n = len(board)
        

    def copy(self):
        return Board(board=torch.clone(self.board))
    
    def display(self, axes=False):
        if axes:
            plain = self.display().split("\n")
            top = " ".join(chr(97 + i) for i in range(self.n))
            left = [" "] + list(map(str, range(1, self.n + 1)))
            return "\n".join([f"{a} {b}" for a, b in zip(left, [top] + plain)])

        result = []
        # 對於 human player 可走的路徑
        legal_moves = self.all_legal_moves()

        def char(i, j):
            if self.board[i, j] == 0:
                if legal_moves[i, j]:
                    return f"{Fore.GREEN}v{Style.RESET_ALL}"
                else:
                    return "·"
            elif self.board[i, j] == self.board_disp_ij: # 人類先手要設為 -1，人類後手要設為 1
                return f"{Fore.RED}o{Style.RESET_ALL}"
            else:  # self.board[i, j] == -1 先手x
                return f"{Fore.BLUE}x{Style.RESET_ALL}"
            
        for i in range(self.n):
            row = []
            for j in range(self.n):
                row.append(char(i, j))
            result.append(" ".join(row))

        return "\n".join(result)

        # return "\n".join(" ".join(char(c) for c in row) for row in self.board)

    __repr__ = __str__ = display

    def rep(self):
        return (
            (self.board.flatten() == 1).long().dot(self.POW2).item(),
            (self.board.flatten() == -1).long().dot(self.POW2).item(),
        )

    def is_move_legal(self, x, y, p=1):
        if not (0 <= x < self.n and 0 <= y < self.n):
            return False

        if self.board[x, y] != 0:
            return False

        for dx, dy in self.DIRECTIONS:
            for n in range(1, 8):
                xp, yp = x + n * dx, y + n * dy
                if not (0 <= xp < self.n and 0 <= yp < self.n):
                    break
                b = self.board[x + n * dx, y + n * dy]
                if b == 0 or (n == 1 and b != -p):
                    break
                if n > 1 and b == p:
                    return True

    def all_legal_moves(self, p=1):
        ind = torch.zeros((self.n, self.n), dtype=torch.bool)
        for i, j in it.product(*[range(self.n)] * 2):
            if self.is_move_legal(i, j, p):
                ind[i, j] = 1
        return ind

    def pi_mask(self, p=1):
        moves = self.all_legal_moves(p)
        may_pass = moves.sum().item() == 0
        return torch.cat([moves.flatten(), torch.tensor([may_pass])])

    def play(self, x, y, p=1):
        assert self.is_move_legal(x, y, p)

        self.board[x, y] = p

        for dx, dy in self.DIRECTIONS:
            for n in range(1, 8):
                xp, yp = x + n * dx, y + n * dy
                if not (0 <= xp < self.n and 0 <= yp < self.n):
                    break
                b = self.board[x + n * dx, y + n * dy]
                if b == 0 or (n == 1 and b != -p):
                    break
                if n > 1 and b == p:
                    for k in range(1, n + 1):
                        xp, yp = x + k * dx, y + k * dy
                        self.board[xp, yp] = p
                    break

    def move(self, i, p=1):
        assert 0 <= i < self.n ** 2 + 1
        if i == self.n ** 2:
            return

        x, y = divmod(i, self.n)
        self.play(x, y, p)

    def flip(self):
        self.board *= -1

    def has_moves(self, p=1):
        ind = torch.zeros((self.n, self.n))
        for i, j in it.product(*[range(self.n)] * 2):
            if self.is_move_legal(i, j, p):
                return True
        return False

    def reward(self, p=1):
        """
        - p=1 代表玩家1,也就是先手方。而p=-1則代表玩家2, 也就是後手方
        - 先手方的棋子數量比後手方多, 後手方獲勝
        - 先手方的棋子以 1 表示, 後守方已 -1 表示，將盤面所有棋子相加，若為正，則後手方獲勝，反之則先手方獲勝
        """
        if p * self.board.sum() < 0: 
            return 1
        elif p * self.board.sum() > 0:
            return -1
        elif self.board.sum() == 0:
            nonzero = float(torch.rand(1) - 0.5) / 100
            assert nonzero != 0
            return nonzero

    def get_game_ended(self, p=1):
        if self.has_moves(p):
            return 0
        if self.has_moves(-p):
            return 0
        return -self.reward(p)

    def get_symmetries(self, pi):
        assert len(pi) == self.n ** 2 + 1
        pi_2d = pi[:-1].view(self.n, self.n)
        results = []

        for f, r in it.product(
            (lambda b: b, lambda b: torch.flip(b, [0])),
            map(lambda k: lambda b: torch.rot90(b, k), range(4)),
        ):
            new_self = Board(board=r(f(self.board)))
            new_pi = torch.cat((r(f(pi_2d)).flatten(), pi[-1].ravel()))
            results.append((new_self, new_pi))

        return results