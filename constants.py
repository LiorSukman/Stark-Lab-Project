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
