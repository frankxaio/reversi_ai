CONFIG = {
    'kill_action': 30,      #和棋回合數
    'dirichlet': 0.35,       # 國際象棋，0.3；日本將棋，0.15；圍棋，0.03，走子選擇越多係數越小
    'play_out': 1200,        # 每次移動的模擬次數
    'c_puct': 5,             # u的權重
    'buffer_size': 100000,   # 經驗池大小
    'paddle_model_path': 'current_policy.model',      # paddle模型路徑
    'pytorch_model_path': 'current_policy.pkl',   # pytorch模型路徑
    'train_data_buffer_path': 'train_data_buffer.pkl',   # 數據容器的路徑
    'batch_size': 512,  # 每次更新的train_step數量
    'kl_targ': 0.02,  # kl散度控制
    'epochs' : 5,  # 每次更新的train_step數量
    'game_batch_num': 3000,  # 訓練更新的次數
    'use_frame': 'pytorch',  # paddle or pytorch根據自己的環境進行切換
    'train_update_interval': 600,  #模型更新間隔時間
    'use_redis': False, # 數據存儲方式
    'redis_host': 'localhost',
    'redis_port': 6379,
    'redis_db': 0,
}