import az, torch
from pathlib import Path

P = az.PolicyBig().cuda()
P.load_state_dict(torch.load("model/iter00088.pt"))

board = az.Board()
MP = az.MCTSPlayer(P, 0, 0, 3, 500)
MP.init()
A = az.Arena(MP, az.HumanPlayer())

# 玩家對戰要在 Arena.play 開啟 print(action)
# AI 執 "X" 先手，將 self.board_disp_ij = 1
#result = A.play(1) # 設定AI先手(1)還是後手(-1)

# AI 執 "O" 後手，將 self.board_disp_ij = -1
result = A.play(-1)

# if result == -1:
#  print("you win!")
# else:
#  print("you lose!")