# -------- training parameters -----------
EPOCHS = 100
LEARNING_RATE = 0.005
BATCH_SIZE = 16
K_FOLDS = 10
HIDDEN_CHANNELS = 8
HEADS = 8
DROPOUT = 0.2
ROOT = './data_control/'
DATASET_NAME = 'MUTAG'

# --------- analysis parameters ----------
DISTANCE_MODE = "cost"  # else edit_step/num_all_ops is distance measure
CORRECTLY_CLASSIFIED_ONLY = True  # to only consider paths with correctly classified source and target
FULLY_CONNECTED_ONLY = True  # to only consider edit path graphs which are fully connected, hence plausible

# ----------- merged dataset parameters ----------
LABEL_MODE = "same_only"
