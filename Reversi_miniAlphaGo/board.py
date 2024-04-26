#!/usr/bin/Anaconda3/python
# -*- coding: utf-8 -*-

from rich import print
# from rich.console import Console
import copy
import numpy as np

class Board(object):
    """
    Board 黑白棋棋盤，規格是8*8，黑棋用 X 表示，白棋用 O 表示，未落子時用 . 表示。
    """

    def __init__(self):
        """
        初始化棋盤狀態
        """
        self.empty = '.'  # 未落子狀態
        self._board = [[self.empty for _ in range(8)] for _ in range(8)]  # 規格：8*8
        self._board[3][3] = 'X'  # 黑棋棋子
        self._board[4][4] = 'X'  # 黑棋棋子
        self._board[3][4], self._board[4][3] = 'O', 'O'  # 白棋棋子

    def __getitem__(self, index):
        """
        添加Board[][] 索引語法
        :param index: 下標索引
        :return:
        """
        return self._board[index]

    def display(self, step_time=None, total_time=None):
        """
        打印棋盤
        :param step_time: 每一步的耗時, 比如:{"X":1,"O":0},默認值是None
        :param total_time: 總耗時, 比如:{"X":1,"O":0},默認值是None
        :return:
        """
        board_print = copy.deepcopy(self._board)
        for i in range(8):
            for j in range(8):
                if 'X' == board_print[i][j]:
                    board_print[i][j] = '[bold blue]X[/bold blue]'
                if 'O' == board_print[i][j]:
                    board_print[i][j] = '[red]O[/red]'
        # print(step_time,total_time)
        # 打印列名
        print(' ', ' '.join(list('ABCDEFGH')))
        # 打印行名和棋盤
        for i in range(8):
            # print(board)
            print('[white]'+str(i + 1), ' '.join(board_print[i]))
        if (not step_time) or (not total_time):
            # 棋盤初始化時展示的時間
            step_time = {"X": 0, "O": 0}
            total_time = {"X": 0, "O": 0}
            print("統計棋局: 棋子總數 / 每一步耗時 / 總時間 ")
            print("黑   棋: " + str(self.count('X')) + ' / ' + str(step_time['X']) + ' / ' + str(
                total_time['X']))
            print("白   棋: " + str(self.count('O')) + ' / ' + str(step_time['O']) + ' / ' + str(
                total_time['O']) + '\n')
        else:
            # 比賽時展示時間
            print("統計棋局: 棋子總數 / 每一步耗時 / 總時間 ")
            print("黑   棋: " + str(self.count('X')) + ' / ' + str(step_time['X']) + ' / ' + str(
                total_time['X']))
            print("白   棋: " + str(self.count('O')) + ' / ' + str(step_time['O']) + ' / ' + str(
                total_time['O']) + '\n')

    def count(self, color):
        """
        統計 color 一方棋子的數量。(O:白棋, X:黑棋, .:未落子狀態)
        :param color: [O,X,.] 表示棋盤上不同的棋子
        :return: 返回 color 棋子在棋盤上的總數
        """
        count = 0
        for y in range(8):
            for x in range(8):
                if self._board[x][y] == color:
                    count += 1
        return count

    def get_winner(self):
        """
        判斷黑棋和白旗的輸贏，通過棋子的個數進行判斷
        :return: 0-黑棋贏, 1-白旗贏, 2-表示平局，黑棋個數和白旗個數相等
        """
        # 定義黑白棋子初始的個數
        black_count, white_count = 0, 0
        for i in range(8):
            for j in range(8):
                # 統計黑棋棋子的個數
                if self._board[i][j] == 'X':
                    black_count += 1
                # 統計白旗棋子的個數
                if self._board[i][j] == 'O':
                    white_count += 1
        if black_count < white_count:
            # 黑棋勝
            return 0, black_count - white_count
        elif black_count > white_count:
            # 白棋勝
            return 1, white_count - black_count
        elif black_count == white_count:
            # 表示平局，黑棋個數和白旗個數相等
            return 2

    def _move(self, action, color):
        """
        落子並獲取反轉棋子的坐標
        :param action: 落子的坐標若是 D3 將他轉成 (2,3)
        :param color: [O,X,.] 表示棋盤上不同的棋子
        :return: 返回反轉棋子的坐標列表, 落子失敗則返回 False
        """
        # 判斷action 是不是字符串，如果是則轉化為數字坐標
        if isinstance(action, str):
            action = self.board_num(action)

        fliped = self._can_fliped(action, color)

        if fliped:
            # 有就反轉對方棋子坐標
            for flip in fliped:
                x, y = self.board_num(flip)
                self._board[x][y] = color

            # 落子坐標
            x, y = action
            # 更改棋盤上 action 坐標處的狀態，修改之後該位置屬於 color[X,O,.]等三狀態
            self._board[x][y] = color
            return fliped
        else:
            # 沒有反轉子則落子失敗
            return False

    def backpropagation(self, action, flipped_pos, color):
        """
        回溯
        :param action: 落子點的坐標
        :param flipped_pos: 反轉棋子坐標列表
        :param color: 棋子的屬性，[X,0,.]三種情況
        :return:
        """
        # 判斷action 是不是字符串，如果是則轉化為數字坐標
        if isinstance(action, str):
            action = self.board_num(action)

        self._board[action[0]][action[1]] = self.empty
        # 如果 color == 'X'，則 op_color = 'O';否則 op_color = 'X'
        op_color = "O" if color == "X" else "X"

        for p in flipped_pos:
            # 判斷action 是不是字符串，如果是則轉化為數字坐標
            if isinstance(p, str):
                p = self.board_num(p)
            self._board[p[0]][p[1]] = op_color

    def is_on_board(self, x, y):
        """
        判斷坐標是否出界
        :param x: row 行坐標
        :param y: col 列坐標
        :return: True or False
        """
        return x >= 0 and x <= 7 and y >= 0 and y <= 7

    def _can_fliped(self, action, color):
        """
        檢測落子是否合法,如果不合法，返回 False,否則返回反轉子的坐標列表
        :param action: 下子位置
        :param color: [X,0,.] 棋子狀態
        :return: False or 反轉對方棋子的坐標列表
        """
        # 判斷action 是不是字符串，如果是則轉化為數字坐標
        if isinstance(action, str):
            action = self.board_num(action)
        xstart, ystart = action

        # 如果該位置已經有棋子或者出界，返回 False
        if not self.is_on_board(xstart, ystart) or self._board[xstart][ystart] != self.empty:
            return False

        # 臨時將color放到指定位置
        self._board[xstart][ystart] = color
        # 棋手
        op_color = "O" if color == "X" else "X"

        # 要被翻轉的棋子
        flipped_pos = []
        flipped_pos_board = []

        for xdirection, ydirection in [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0],
                                       [-1, 1]]:
            x, y = xstart, ystart
            x += xdirection
            y += ydirection
            # 如果(x,y)在棋盤上，而且為對方棋子,則在這個方向上繼續前進，否則循環下一個角度。
            if self.is_on_board(x, y) and self._board[x][y] == op_color:
                x += xdirection
                y += ydirection
                # 進一步判斷點(x,y)是否在棋盤上，如果不在棋盤上，繼續循環下一個角度,如果在棋盤上，則進行while循環。
                if not self.is_on_board(x, y):
                    continue
                # 一直走到出界或不是對方棋子的位置
                while self._board[x][y] == op_color:
                    # 如果一直是對方的棋子，則點（x,y）一直循環，直至點（x,y)出界或者不是對方的棋子。
                    x += xdirection
                    y += ydirection
                    # 點(x,y)出界了和不是對方棋子
                    if not self.is_on_board(x, y):
                        break
                # 出界了，則沒有棋子要翻轉OXXXXX
                if not self.is_on_board(x, y):
                    continue

                # 是自己的棋子OXXXXXXO
                if self._board[x][y] == color:
                    while True:
                        x -= xdirection
                        y -= ydirection
                        # 回到了起點則結束
                        if x == xstart and y == ystart:
                            break
                        # 需要翻轉的棋子
                        flipped_pos.append([x, y])

        # 將前面臨時放上的棋子去掉，即還原棋盤
        self._board[xstart][ystart] = self.empty  # restore the empty space

        # 沒有要被翻轉的棋子，則走法非法。返回 False
        if len(flipped_pos) == 0:
            return False

        for fp in flipped_pos:
            flipped_pos_board.append(self.num_board(fp))
        # 走法正常，返回翻轉棋子的棋盤坐標
        return flipped_pos_board

    def get_legal_actions(self, color):
        """
        按照黑白棋的規則獲取棋子的合法走法
        :param color: 不同顏色的棋子, X-黑棋, O-白棋
        :return: 生成合法的落子坐標, 用list()方法可以獲取所有的合法坐標
        """
        # 表示棋盤坐標點的8個不同方向坐標，比如方向坐標[0][1]則表示坐標點的正上方。
        direction = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

        op_color = "O" if color == "X" else "X"
        # 統計 op_color 一方鄰近的未落子狀態的位置
        op_color_near_points = []

        board = self._board
        for i in range(8):
            # i 是行數，從0開始，j是列數，也是從0開始
            for j in range(8):
                # 判斷棋盤[i][j]位子棋子的屬性，如果是op_color，則繼續進行下一步操作，
                # 否則繼續循環獲取下一個坐標棋子的屬性
                if board[i][j] == op_color:
                    # dx，dy 分別表示[i][j]坐標在行、列方向上的步長，direction 表示方向坐標
                    for dx, dy in direction:
                        x, y = i + dx, j + dy
                        # 表示x、y坐標值在合理範圍，棋盤坐標點board[x][y]為未落子狀態，
                        # 而且（x,y）不在op_color_near_points 中，統計對方未落子狀態位置的列表才可以添加該坐標點
                        if 0 <= x <= 7 and 0 <= y <= 7 and board[x][y] == self.empty and (
                                x, y) not in op_color_near_points:
                            op_color_near_points.append((x, y))
        l = [0, 1, 2, 3, 4, 5, 6, 7]
        for p in op_color_near_points:
            if self._can_fliped(p, color):
                # 判斷p是不是數字坐標，如果是則返回棋盤坐標
                # p = self.board_num(p)
                if p[0] in l and p[1] in l:
                    p = self.num_board(p)
                yield p

    def board_num(self, action):
        """
        棋盤坐標轉化為數字坐標
        :param action:棋盤坐標，比如A1
        :return:數字坐標，比如 A1 --->(0,0)
        """
        row, col = str(action[1]).upper(), str(action[0]).upper()
        if row in '12345678' and col in 'ABCDEFGH':
            # 坐標正確
            x, y = '12345678'.index(row), 'ABCDEFGH'.index(col)
            return x, y

    def num_board(self, action):
        """
        數字坐標轉化為棋盤坐標
        :param action:數字坐標 ,比如(0,0)
        :return:棋盤坐標，比如 (0,0)---> A1
        """
        row, col = action
        l = [0, 1, 2, 3, 4, 5, 6, 7]
        if col in l and row in l:
            return chr(ord('A') + col) + str(row + 1)

    def one_hot_encoding(self):
        """
        將棋盤的資訊轉換成 one-hot encoding 表示，也就是將棋盤上表示成寬 8 長 8 深度為 3 的矩陣。深度也就是 channel.
        X --> [0,1,0], O --> [0,0,1], . --> [1,0,0]
        channel 0: 無子狀態, channel 1: 黑子狀態, channel 2: 白子狀態
        :return: one-hot encoding 表示的棋盤
        """
        board = self._board
        board_one_hot = copy.deepcopy(board)
        string2array = {".": np.array([1, 0, 0]), "X": np.array([0, 1, 0]), "O": np.array([0, 0, 1])}
        for i in range(8):
            for j in range(8):
                board_one_hot[i][j] = string2array[board[i][j]]
        board_one_hot = np.array(board_one_hot)
        transposed_one_hot = np.transpose(board_one_hot, (2, 0, 1)) # 原本是 (8, 8, 3) 轉換成 (3, 8, 8)
        # print(transposed_one_hot)
        board_one_hot = [arr.tolist() for arr in transposed_one_hot]
        return board_one_hot

    @property
    def board(self):
        # 任何使用到 self._board 的地方都會直接修改 self._board 這個棋盤，若要複製一模一樣的棋盤要使用 deepcopy
        return self._board

# # 測試
if __name__ == '__main__':
    board = Board()  # 棋盤初始化
    # print(board[0][0])
    # 印出棋盤
    board.display()

    # one hot encoding 表示棋盤測試
    # one_hot = board.one_hot_encoding()
    # print(one_hot) # 檢查是不是 list
    # print(np.array(one_hot).shape) # 確認維度是否為 (3, 8, 8)

    # board move 落子測試
    # board._move("D6",'X')
    # board.display()

    # get_winnter 測試
    # get_winner = board.get_winner()
    # print(get_winner)

    # get_legal_actions 測試
    legal_actions = list(board.get_legal_actions('X'))
    print(legal_actions)