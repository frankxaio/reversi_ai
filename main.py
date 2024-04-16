import pygame, sys, random, time
from pygame.locals import *
from tool import *
from AI import *
from gui import *

# GUI
BACKGROUNDCOLOR = (255, 255, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 22)
CELLWIDTH = 50
CELLHEIGHT = 50
PIECEWIDTH = 44
PIECEHEIGHT = 44
BOARDX = 0
BOARDY = 0
FPS = 40
game_window_extend = 100
textMargin = 10
textSpacing = 30


if __name__ == '__main__':
    # initialize
    pygame.init()

    # load image
    boardImage = pygame.image.load('board.png')
    boardRect = boardImage.get_rect()
    blackImage = pygame.image.load('black.png')
    blackRect = blackImage.get_rect()
    whiteImage = pygame.image.load('white.png')
    whiteRect = whiteImage.get_rect()

    basicFont = pygame.font.SysFont(None, 48)
    gameoverStr = 'Game Over Score '

    mainBoard = getNewBoard()
    resetBoard(mainBoard)

    turn = whoGoesFirst()
    if turn == 'player':
        playerTile = 'black'
        computerTile = 'white'
    else:
        playerTile = 'white'
        computerTile = 'black'

    print(turn)

    # 遊戲視窗
    windowSurface = pygame.display.set_mode((boardRect.width+game_window_extend, boardRect.height))
    pygame.display.set_caption('黑白棋')

    # 初始化:
    gameOver = False
    start_player = time.time()
    total_player_time = 0
    total_computer_time = 0
    has_executed = False

    # 遊戲主循環
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                terminate()
            # 玩家下棋
            if isGameOver(mainBoard) == False and turn == 'player' and event.type == MOUSEBUTTONDOWN and event.button == 1:
                x, y = pygame.mouse.get_pos()
                end_player = time.time()
                timer_player = end_player - start_player
                total_player_time += timer_player
                print(f"Player takes : {timer_player:.2f}s")
                # 滑鼠點擊對應到棋盤的位置
                col = int((x - BOARDX) / CELLWIDTH)
                row = int((y - BOARDY) / CELLHEIGHT)
                if makeMove(mainBoard, playerTile, col, row) == True:
                    if getValidMoves(mainBoard, computerTile) != []:
                        turn = 'computer'
            if event.type == KEYUP:
                if event.key == K_q:
                    turn = 'computer'

        # 顯示棋盤
        windowSurface.fill(BACKGROUNDCOLOR)
        windowSurface.blit(boardImage, boardRect, boardRect)

        # 電腦下棋
        if (isGameOver(mainBoard) == False and turn == 'computer'):
            x, y, timer = getComputerMove(mainBoard, computerTile)
            makeMove(mainBoard, computerTile, x, y)
            savex, savey = x, y
            total_computer_time += total_computer_time
            print(f'AI takes: {timer:.4f} seconds')
            start_player = time.time()

            # 玩家没有可行的走法了
            if getValidMoves(mainBoard, playerTile) != []:
                turn = 'player'

        score = getScoreOfBoard(mainBoard)
        if score['black'] == 0 or score['white'] == 0 or isGameOver(mainBoard):
            gameOver = True

        windowSurface.fill(BACKGROUNDCOLOR)
        windowSurface.blit(boardImage, boardRect, boardRect)

        # 顯示記分板
        ScoreBoard(mainBoard, windowSurface)

        validMoves = getValidMoves(mainBoard, playerTile)

        # 在介面畫出棋子
        for x in range(8):
            for y in range(8):
                # 計算標記的位置
                rectDst = pygame.Rect(BOARDX + x * CELLWIDTH + 2, BOARDY + y * CELLHEIGHT + 2, PIECEWIDTH, PIECEHEIGHT)
                if mainBoard[x][y] == 'black':
                    windowSurface.blit(blackImage, rectDst, blackRect)
                elif mainBoard[x][y] == 'white':
                    windowSurface.blit(whiteImage, rectDst, whiteRect)
                # 繪製合法路徑
                if [x, y] in validMoves:
                    pygame.draw.circle(windowSurface, (136,196,255), rectDst.center, 18, 3)


        # 檢查遊戲結束
        if isGameOver(mainBoard) == True:
            scorePlayer = getScoreOfBoard(mainBoard)[playerTile]
            scoreComputer = getScoreOfBoard(mainBoard)[computerTile]
            if scorePlayer < scoreComputer:
                outputStr = "Win! " + str(scorePlayer) + ":" + str(scoreComputer)
            elif scorePlayer == scoreComputer:
                outputStr = "Tie. " + str(scorePlayer) + ":" + str(scoreComputer)
            else:
                outputStr = "Loss. " + str(scorePlayer) + ":" + str(scoreComputer)
            text = basicFont.render(outputStr, True, BLACK, YELLOW)
            textRect = text.get_rect()
            textRect.centerx = windowSurface.get_rect().centerx
            textRect.centery = windowSurface.get_rect().centery
            windowSurface.blit(text, textRect)

            # 印出總時間
            if not has_executed:
                # 執行需要只執行一次的代碼
                # print("這段代碼只會執行一次")
                # 設定已執行過的 flag
                has_executed = True
                print(f"Total player time: {total_player_time:.2f} seconds")
                print(f"Total computer time: {total_computer_time:.2f} seconds")

        pygame.display.update()
