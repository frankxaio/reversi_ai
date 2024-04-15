import random

# 重製棋盤
def resetBoard(board):
    for x in range(8):
        for y in range(8):
            board[x][y] = 'none'

    # Starting pieces:
    board[3][3] = 'black'
    board[3][4] = 'white'
    board[4][3] = 'white'
    board[4][4] = 'black'


# 建立新棋盤
def getNewBoard():
    board = []
    for i in range(8):
        board.append(['none'] * 8)

    return board


# 是否為合法走法
def isValidMove(board, tile, xstart, ystart):
    # 如果該位置已經有棋子或者出界了，返回False
    if not isOnBoard(xstart, ystart) or board[xstart][ystart] != 'none':
        return False

    # 臨時將tile 放到指定的位置
    board[xstart][ystart] = tile

    if tile == 'black':
        otherTile = 'white'
    else:
        otherTile = 'black'

    # 要被翻轉的棋子
    tilesToFlip = []
    for xdirection, ydirection in [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]:
        x, y = xstart, ystart
        x += xdirection
        y += ydirection
        if isOnBoard(x, y) and board[x][y] == otherTile:
            x += xdirection
            y += ydirection
            if not isOnBoard(x, y):
                continue
            # 一直走到出界或不是對方棋子的位置
            while board[x][y] == otherTile:
                x += xdirection
                y += ydirection
                if not isOnBoard(x, y):
                    break
            # 出界了，則沒有棋子要翻轉OXXXXX
            if not isOnBoard(x, y):
                continue
            # 是自己的棋子OXXXXXXO
            if board[x][y] == tile:
                while True:
                    x -= xdirection
                    y -= ydirection
                    # 回到了起點則結束
                    if x == xstart and y == ystart:
                        break
                    # 需要翻轉的棋子
                    tilesToFlip.append([x, y])

    # 將前面臨時放上的棋子去掉，即還原棋盤
    board[xstart][ystart] = 'none'  # restore the empty space

    # 沒有要被翻轉的棋子，則走法非法。翻轉棋的規則。
    if len(tilesToFlip) == 0:  # If no tiles were flipped, this is not a valid move.
        return False
    return tilesToFlip


# 是否出界
def isOnBoard(x, y):
    return x >= 0 and x <= 7 and y >= 0 and y <= 7


# 獲取合法位置
def getValidMoves(board, tile):
    validMoves = []

    for x in range(8):
        for y in range(8):
            if isValidMove(board, tile, x, y) != False:
                validMoves.append([x, y])
    return validMoves


# 棋盤雙方黑白子的分數
def getScoreOfBoard(board):
    xscore = 0
    oscore = 0
    for x in range(8):
        for y in range(8):
            if board[x][y] == 'black':
                xscore += 1
            if board[x][y] == 'white':
                oscore += 1
    return {'black': xscore, 'white': oscore}


# 隨機決定誰先走
def whoGoesFirst():
    if random.randint(0, 1) == 0:
        return 'computer'
    else:
        return 'player'


# 將一個tile旗子放到(xstart, ystart)
def makeMove(board, tile, xstart, ystart):
    tilesToFlip = isValidMove(board, tile, xstart, ystart)

    if tilesToFlip == False:
        return False

    board[xstart][ystart] = tile
    for x, y in tilesToFlip:
        board[x][y] = tile
    return True


# 複製棋盤
def getBoardCopy(board):
    dupeBoard = getNewBoard()

    for x in range(8):
        for y in range(8):
            dupeBoard[x][y] = board[x][y]

    return dupeBoard


# 是否在角上
def isOnCorner(x, y):
    return (x == 0 and y == 0) or (x == 7 and y == 0) or (x == 0 and y == 7) or (x == 7 and y == 7)

# 是否遊戲結束
def isGameOver(board):
    score = getScoreOfBoard(board)
    if score['black'] == 0 or score['white'] == 0:
        return True

    for x in range(8):
        for y in range(8):
            if board[x][y] == 'none':
                return False
    return True
