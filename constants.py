import numpy as np

NUM_CHANNELS = 8
TIMESTEPS = 32
SAMPLE_RATE = 20_000
UPSAMPLE = 8
# be careful and assert it is not being changed
COORDINATES = np.array([[0, 0], [-9, 20], [8, 40], [-13, 60], [12, 80], [-17, 100], [16, 120], [-21, 140]])
MIN_TIME_LIST = 8000
VERBOS = False
SEED = 0
INF = 9999
SPATIAL = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, -1]
MORPHOLOGICAL = [15, 16, 17, 18, 19, 20, 21, 22, -1]
TEMPORAL = [23, 24, 25, 26, 27, 28, -1]
SPAT_TEMPO = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 23, 24, 25, 26, 27, 28, -1]
