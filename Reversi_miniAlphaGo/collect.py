"""自我對弈收集數據"""
import random
from collections import deque
import copy
import os
import pickle
import time
# from game import Board, Game, move_action2move_id, move_id2move_action, flip_map
from game import Game
from board import Board
from mcts_rl import MCTSPlayer
from config import CONFIG
from model import PolicyValueNet

if CONFIG['use_redis']:
    import my_redis, redis

# TODO 寫 zip_array，製造翻轉數據
# import zip_array
#
# if CONFIG['use_frame'] == 'paddle':
#     from paddle_net import PolicyValueNet
# elif CONFIG['use_frame'] == 'pytorch':
#     from pytorch_net import PolicyValueNet
# else:
#     print('暫不支持您選擇的框架')


# 定義整個對弈收集數據流程
class CollectPipeline:

    def __init__(self, init_model=None):
        # 象棋邏輯和棋盤
        self.board = Board()
        self.game = Game()
        # 對弈參數
        self.temp = 1  # 溫度
        self.n_playout = CONFIG['play_out']  # 每次移動的模擬次數
        self.c_puct = CONFIG['c_puct']  # u的權重
        self.buffer_size = CONFIG['buffer_size']  # 經驗池大小
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.iters = 0
        if CONFIG['use_redis']:
            self.redis_cli = my_redis.get_redis_cli()

    # 從主體加載模型
    def load_model(self):
        if CONFIG['use_frame'] == 'paddle':
            model_path = CONFIG['paddle_model_path']
        elif CONFIG['use_frame'] == 'pytorch':
            model_path = CONFIG['pytorch_model_path']
        else:
            print('暫不支持所選框架')
        try:
            self.policy_value_net = PolicyValueNet(model_file=model_path)
            print('已加載最新模型')
        except:
            self.policy_value_net = PolicyValueNet()
            print('已加載初始模型')
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)

    # def get_equi_data(self, play_data):
    #     """左右對稱變換，擴充數據集一倍，加速一倍訓練速度"""
    #     extend_data = []
    #     # 棋盤狀態shape is [9, 10, 9], 走子概率，贏家
    #     for state, mcts_prob, winner in play_data:
    #         # 原始數據
    #         extend_data.append(zip_array.zip_state_mcts_prob((state, mcts_prob, winner)))
    #         # 水平翻轉後的數據
    #         state_flip = state.transpose([1, 2, 0])
    #         state = state.transpose([1, 2, 0])
    #         for i in range(10):
    #             for j in range(9):
    #                 state_flip[i][j] = state[i][8 - j]
    #         state_flip = state_flip.transpose([2, 0, 1])
    #         mcts_prob_flip = copy.deepcopy(mcts_prob)
    #         for i in range(len(mcts_prob_flip)):
    #             mcts_prob_flip[i] = mcts_prob[move_action2move_id[flip_map(move_id2move_action[i])]]
    #         extend_data.append(zip_array.zip_state_mcts_prob((state_flip, mcts_prob_flip, winner)))
    #     return extend_data

    def collect_selfplay_data(self, n_games=1):
        # 收集自我對弈的數據
        for i in range(n_games):
            self.load_model()  # 從本體處加載最新模型
            # TODO 寫自我對函數
            winner, play_data = self.game.start_self_play(self.mcts_player, temp=self.temp, is_shown=False)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            # TODO 寫翻轉數據
            # play_data = self.get_equi_data(play_data)
            if CONFIG['use_redis']:
                while True:
                    try:
                        for d in play_data:
                            self.redis_cli.rpush('train_data_buffer', pickle.dumps(d))
                        self.redis_cli.incr('iters')
                        self.iters = self.redis_cli.get('iters')
                        print("存儲完成")
                        break
                    except:
                        print("存儲失敗")
                        time.sleep(1)
            else:
                if os.path.exists(CONFIG['train_data_buffer_path']):
                    while True:
                        try:
                            with open(CONFIG['train_data_buffer_path'], 'rb') as data_dict:
                                data_file = pickle.load(data_dict)
                                self.data_buffer = deque(maxlen=self.buffer_size)
                                self.data_buffer.extend(data_file['data_buffer'])
                                self.iters = data_file['iters']
                                del data_file
                                self.iters += 1
                                self.data_buffer.extend(play_data)
                            print('成功載入數據')
                            break
                        except:
                            time.sleep(30)
                else:
                    self.data_buffer.extend(play_data)
                    self.iters += 1
            data_dict = {'data_buffer': self.data_buffer, 'iters': self.iters}
            with open(CONFIG['train_data_buffer_path'], 'wb') as data_file:
                pickle.dump(data_dict, data_file)
        return self.iters

    def run(self):
        """開始收集數據"""
        try:
            while True:
                iters = self.collect_selfplay_data()
                print('batch i: {}, episode_len: {}'.format(
                    iters, self.episode_len))
        except KeyboardInterrupt:
            print('\n\rquit')


collecting_pipeline = CollectPipeline(init_model='current_policy.model')
collecting_pipeline.run()

if CONFIG['use_frame'] == 'paddle':
    collecting_pipeline = CollectPipeline(init_model='current_policy.model')
    collecting_pipeline.run()
elif CONFIG['use_frame'] == 'pytorch':
    collecting_pipeline = CollectPipeline(init_model='current_policy.pkl')
    collecting_pipeline.run()
else:
    print('暫不支持您選擇的框架')
    print('訓練結束')