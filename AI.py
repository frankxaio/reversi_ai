from tool import *

def getComputerMove(board, computerTile):
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
    return bestMove

# # 電腦走法，AI, rewrite
# def getComputerMove(board, computerTile):
#
#     possibleMoves = getValidMoves(board, computerTile) # 获取所有合法走法
#     print(possibleMoves)  # for test
#     predict = []
#     for pos in possibleMoves: # 对所有走法进行尝试，用蒙特卡洛树算法进行搜索，最后选择模拟胜率最高的一个走法
#         MCTree = MCTS(board, pos, computerTile)  # MCTree 是蒙特卡洛树的一个对象， 输入当前棋盘和打算下棋位置
#         predict.append(MCTree.evaluate())  # 返回一个（位置，胜率）
#
#     # 从预测列表中选出一个胜率最大的位置
#     return predict[0]