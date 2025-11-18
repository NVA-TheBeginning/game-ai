# TODO: handle these settings via command-line args or config file
MODE = "bot"  # "bot" or "interface"
DEBUG_MODE = False

SERVER_WS = "ws://localhost:3000/bot"
INTERFACE_PORT = 8765
QTABLE_FILE = "qtable.pkl"

PRUNE_MAX_TARGETS = 50
ATTACK_RATIOS = [0.20, 0.40, 0.60, 0.80]
EPSILON_MIN = 0.02
EPSILON_DECAY = 0.9995

AUTOSAVE_INTERVAL = 300
PRINT_INTERVAL = 20

REWARD_SPAWN_SUCCESS = 150.0
REWARD_MISSED_SPAWN = -150.0
REWARD_SMALL_STEP = -0.1
