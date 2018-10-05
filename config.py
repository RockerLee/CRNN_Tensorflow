
DEBUG = True

# Supported characters
CHAR_VECTOR = "0123456789+-*()="

# Number of classes: CHAR_VECTOR and ctc blank
NUM_CLASSES = len(CHAR_VECTOR) + 1

# split train & test, random seed
RANDOM_STATE=42

# max train epochs
EPOCHS = 40000

BATCH_SIZE = 128

# learning rate
INITIAL_LEARNING_RATE = 1e-3
DECAY_STEPS = 10000
DECAY_RATE = 0.1
STAIRCASE = False

REPORT_STEPS = 10

MODEL_PATH = 'checkpoint/sample_1w'
PATH_TBOARD_TRAIN = 'TensorBoard/train_1w'
PATH_TBOARD_TEST = 'TensorBoard/test'