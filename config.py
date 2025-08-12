# configuration parameters

# training parameters
EPOCHS = 100
LEARNING_RATE = 0.005
BATCH_SIZE = 16
K_FOLDS = 10
HIDDEN_CHANNELS = 8
HEADS = 8
DROPOUT = 0.2
ROOT = './data/'
DATASET_NAME = 'MUTAG'

# analysis parameters
DISTANCE_MODE = "cost_function"
CORRECTLY_CLASSIFIED_ONLY = True  # to only consider edit path graphs which are fully connected, hence plausible
FULLY_CONNECTED_ONLY = True  # to only consider paths with correctly classified source and target
