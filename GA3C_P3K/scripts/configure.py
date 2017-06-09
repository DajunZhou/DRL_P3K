
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
    energy = 0
    forward_speed = 0.5
    angular_speed = 1.5
    actions_dim = 3
    episode_time = 60
    EPISODES = 1000000000

    TENSORBOARD = True
    TENSORBOARD_UPDATE_FREQUENCY = 2000
    SAVE_NET = 2000

    #train mode
    mode = 0

    #NET
    DEVICE = 'gpu:0'
    GAMMA = 0.99
    OBSERVE = 10000
    # ANNELING_STEPS = 2000
    INITIAL_EPSILON = 0.0
    FINAL_EPSILON = 0.0
    REPLAY_MEMORY = 80000
    BATCH_SIZE = 20
    REPLAY_START_SIZE = OBSERVE*number_of_bots
    EXPLORE = 8000

    STACKED_FRAMES = 3
    IMAGE_WIDTH = 84
    IMAGE_HEIGHT = 84

    LEARNING_RATE_START = 1e-6
    LEARNING_RATE_END = 1e-6
    TargetNet_Tau = 0.001

    LOAD_CHECKPOINT = True
    SAVE_MODELS = True