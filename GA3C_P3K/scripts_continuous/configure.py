
class configure:

    number_of_food = 30
    number_of_bots = 3
    number_of_pred = 0
    arena_width = 9.0
    number_colour_channels = 3
    number_channels = 4
    laser_scan_range = 2
    number_of_cameras = 1
    camera_size = 30
    min_reward = 0
    max_reward = 3
    energy = 0.0
    allenergy = 1.5
    forward_speed_max = 0.6
    forward_speed_min = 0.0
    angular_speed_max = 1.0
    angular_speed_min = -1.0
    actions_dim = 2
    episode_time = 60
    EPISODES = 1000000000


    TENSORBOARD = True
    TENSORBOARD_UPDATE_FREQUENCY = 1000
    SAVE_NET = 2000

    #train mode
    mode = 0

    #NET
    DEVICE = 'gpu:0'
    LSTM_layer = 1024

    GAMMA = 0.99
    OBSERVE = 25000
    # ANNELING_STEPS = 2000
    INITIAL_EPSILON = 0.0
    FINAL_EPSILON = 0.00
    REPLAY_MEMORY = 80000
    BATCH_SIZE = 20
    REPLAY_START_SIZE = OBSERVE*number_of_bots
    Exp_que_START_SIZE = 100000
    EXPLORE = 100000

    Step = 20
    KEEP_PROB = 1.0

    STACKED_FRAMES = 6
    IMAGE_WIDTH = 84
    IMAGE_HEIGHT = 84

    LEARNING_RATE_START = 1e-6
    LEARNING_RATE_END = 1e-6
    TargetNet_Tau = 0.001

    LOAD_CHECKPOINT = True
    SAVE_MODELS = True

    # A3C
    Lim_Deta = 1.0
    LOG_EPSILON = 1e-6
