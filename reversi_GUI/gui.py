import sys
import pygame
from tool import *

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


# 退出
def terminate():
    pygame.quit()
    sys.exit()

# 記分板
def ScoreBoard(board,windowSurface, total_player_time, total_computer_time):
    boardImage = pygame.image.load('Image/board.png')
    boardRect = boardImage.get_rect()

    blackCount = getScoreOfBoard(board)['black']
    whiteCount = getScoreOfBoard(board)['white']

    # Rendering the count on the window
    smallFont = pygame.font.SysFont(None, 22)
    blackCountText = smallFont.render(f'Black:{blackCount}', True, BLACK)
    whiteCountText = smallFont.render(f'White:{whiteCount}', True, BLACK)
    PlayerTimeText = smallFont.render(f'Timer:{total_player_time:.2f}', True, BLACK)
    ComputerTimeText = smallFont.render(f'Timer:{total_computer_time:.2f}', True, BLACK)


    textMargin = 10
    textSpacing = 30
    # windowSurface.blit(blackCountText, (boardRect.width + textMargin, textMargin))
    # windowSurface.blit(whiteCountText, (boardRect.width + textMargin, textMargin + textSpacing))
    windowSurface.blit(blackCountText, (2, boardRect.height+6))
    windowSurface.blit(whiteCountText, (boardRect.width-70, boardRect.height+6))
    windowSurface.blit(PlayerTimeText, (70, boardRect.height+6))
    windowSurface.blit(ComputerTimeText, (boardRect.width-160, boardRect.height+6))

def ShowStones(mainBoard, windowSurface, validMoves, blackImage, blackRect, whiteImage, whiteRect):
    # 在介面畫出棋子
    for x in range(8):
        for y in range(8):
            # 計算標記的位置
            rectDst = pygame.Rect(BOARDX + x * CELLWIDTH + 2, BOARDY + y * CELLHEIGHT + 2, PIECEWIDTH, PIECEHEIGHT)
            if mainBoard[x][y] == 'black':
                windowSurface.blit(blackImage, rectDst, blackRect)
            elif mainBoard[x][y] == 'white':
                windowSurface.blit(whiteImage, rectDst, whiteRect)
            # 繪製玩家可走路徑
            if [x, y] in validMoves:
                pygame.draw.circle(windowSurface, (136,196,255), rectDst.center, 18, 3)

# def ShowGameOver(board, windowSurface):
#     if isGameOver(board) == True:
#         scorePlayer = getScoreOfBoard(board)[playerTile]
#         scoreComputer = getScoreOfBoard(board)[computerTile]
#         if scorePlayer < scoreComputer:
#             outputStr = "Win! " + str(scorePlayer) + ":" + str(scoreComputer)
#         elif scorePlayer == scoreComputer:
#             outputStr = "Tie. " + str(scorePlayer) + ":" + str(scoreComputer)
#         else:
#             outputStr = "Loss. " + str(scorePlayer) + ":" + str(scoreComputer)
#         text = basicFont.render(outputStr, True, BLACK, YELLOW)
#         textRect = text.get_rect()
#         textRect.centerx = windowSurface.get_rect().centerx
#         textRect.centery = windowSurface.get_rect().centery
#         windowSurface.blit(text, textRect)
