from tool import *
import time
import datetime
from copy import deepcopy


def getComputerMove(board, computerTile):
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
        # 按照當前分數選擇，優先選擇翻轉後分數最少的走法
        score = getScoreOfBoard(dupeBoard)[computerTile]
        if score < bestScore:
            bestMove = [x, y]
            bestScore = score
    end=time.time()
    timer=end-start
    bestMove = [bestMove[0], bestMove[1], timer]
    return bestMove

