from tool import *
import time
import datetime
from copy import deepcopy


def getComputerMove(board, computerTile):
    """
    電腦走法
    :param board: 棋盤用二維陣列表示，`black` 代表黑子，`white` 代表白子，`none` 代表空格。
                [['none', 'none', 'none', 'none', 'none', 'none', 'none', 'none'],
                 ['none', 'none', 'none', 'none', 'none', 'none', 'none', 'none'],
                 ['none', 'none', 'none', 'none', 'none', 'none', 'none', 'none'],
                 ['none', 'none', 'none', 'black', 'white', 'none', 'none', 'none'],
                 ['none', 'none', 'none', 'black', 'black', 'none', 'none', 'none'],
                 ['none', 'none', 'none', 'black', 'none', 'none', 'none', 'none'],
                 ['none', 'none', 'none', 'none', 'none', 'none', 'none', 'none'],
                 ['none', 'none', 'none', 'none', 'none', 'none', 'none', 'none']]
    :param computerTile: 電腦的棋子顏色，`black` 或 `white`
    :return: [x, y, time] 電腦落子的座標和執行時間，左上角為[0,]，右下角為[7, 7]
    """
    start = time.time()
    # 獲取所有合法路徑
    possibleMoves = getValidMoves(board, computerTile)

    # 打亂所有合法走法
    random.shuffle(possibleMoves)

    # [x, y]在角上，則優先走，因為邊角的不會再被翻轉 (在 reverse 的版本不採用)
    # for x, y in possibleMoves:
    #     if isOnCorner(x, y):
    #         return [x, y]

    bestScore = 100000000000
    for x, y in possibleMoves:
        dupeBoard = getBoardCopy(board)
        makeMove(dupeBoard, computerTile, x, y)
        # print(getScoreOfBoard(dupeBoard))
        # 按照當前分數選擇，優先選擇翻轉後分數最少的走法
        score = getScoreOfBoard(dupeBoard)[computerTile]
        if score < bestScore:
            bestMove = [x, y]
            bestScore = score

    end=time.time()
    timer=end-start
    bestMove = [bestMove[0], bestMove[1], timer]
    return bestMove


def minimax(board, tile, depth, isMaximizingPlayer):
    if depth == 0 or isGameOver(board):
        return getScoreOfBoard(board)[tile]

    if isMaximizingPlayer:
        bestScore = float('inf')
        valid_moves = getValidMoves(board, tile)
        for x, y in valid_moves:
            board_copy = getBoardCopy(board)
            makeMove(board_copy, tile, x, y)
            opponent_tile = 'white' if tile == 'black' else 'black'
            score = minimax(board_copy, opponent_tile, depth - 1, False)
            bestScore = min(score, bestScore)
        return bestScore
    else:
        bestScore = float('-inf')
        valid_moves = getValidMoves(board, tile)
        for x, y in valid_moves:
            board_copy = getBoardCopy(board)
            makeMove(board_copy, tile, x, y)
            opponent_tile = 'white' if tile == 'black' else 'black'
            score = minimax(board_copy, opponent_tile, depth - 1, True)
            bestScore = max(score, bestScore)
        return bestScore

def UseMinMax(board, computerTile, depth):
    start = time.time()
    bestScore = float('inf')
    bestMove = None
    valid_moves = getValidMoves(board, computerTile)
    for x, y in valid_moves:
        board_copy = getBoardCopy(board)
        makeMove(board_copy, computerTile, x, y)
        score = minimax(board_copy, computerTile, depth - 1, False)
        if score < bestScore:
            bestScore = score
            bestMove = (x, y)
        if bestMove == None:
            print('NoneType')
            possibleMoves = getValidMoves(board, computerTile)
            random.shuffle(possibleMoves)
            bestMove = possibleMoves[0]
    end=time.time()
    timer=end-start
    return [bestMove[0], bestMove[1], timer]

if __name__ == '__main__':
    board = [['none', 'none', 'none', 'none', 'none', 'none', 'none', 'none'],
             ['none', 'none', 'none', 'none', 'none', 'none', 'none', 'none'],
             ['none', 'none', 'none', 'none', 'black', 'none', 'none', 'none'],
             ['none', 'none', 'none', 'black', 'black', 'none', 'none', 'none'],
             ['none', 'none', 'none', 'white', 'black', 'none', 'none', 'none'],
             ['none', 'none', 'none', 'none', 'none', 'none', 'none', 'none'],
             ['none', 'none', 'none', 'none', 'none', 'none', 'none', 'none'],
             ['none', 'none', 'none', 'none', 'none', 'none', 'none', 'none']]
    computerTile = 'white'
    print(getComputerMove(board, computerTile))
    print(UseMinMax(board, computerTile, 5))