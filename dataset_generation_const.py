IN_DIR = 'waymo_open/training'
OUT_DIR = 'octopus/data/waymo_8_bc/train'
MAX_NUM_TRAIN_SEQ = 40000
SEQ_LEN = 8
STRIDE = 2
BC_STRIDE = 1
BC_LEN = 16
TIMESTAMP_NORMALIZATION_FACTOR = (SEQ_LEN-1)*STRIDE*1e5
DISTANCE_NORMALIZATION_FACTOR = 48 * (1000 / 3600) * (TIMESTAMP_NORMALIZATION_FACTOR*1e-6)
RESIZE_DIM = (224, 168)