import numpy as np

NUM_CHANNELS = 8
TIMESTEPS = 32
SAMPLE_RATE = 20_000
UPSAMPLE = 8

# be careful and assert it is not being changed
COORDINATES = np.array([[0, 0], [-9, 20], [8, 40], [-13, 60], [12, 80], [-17, 100], [16, 120], [-21, 140]])
MIN_TIME_LIST = 8000
ACH_WINDOW = 1000
VERBOS = False
DEBUG = False
SEED = 2
INF = 9999

WIDTH = [14, -1]
T2P = [18, -1]
STARK = [14, 18, -1]
STARK_SPAT_TEMPO = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 18, 21, 22, 23, 24, 25, 26, -1]
WIDTH_SPAT = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, -1]
T2P_SPAT = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, -1]
STARK_SPAT = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 18, -1]

SPATIAL = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, -1]
MORPHOLOGICAL = [13, 14, 15, 16, 17, 18, 19, 20, -1]
TEMPORAL = [22, 23, 24, 25, 26, 27, -1]
SPAT_TEMPO = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 22, 23, 24, 25, 26, 27, -1]
TRANS_MORPH = [0, 1, 2, 3, 4, 5, 6, 7, -1]

PYR_COLOR = (0.416, 0.106, 0.604)
LIGHT_PYR = (0.612, 0.302, 0.8)
PV_COLOR = (0.18, 0.49, 0.196)
LIGHT_PV = (0.376, 0.678, 0.369)
UT_COLOR = (0.082, 0.396, 0.753)
LIGHT_UT = (0.369, 0.573, 0.953)

SESSION_TO_ANIMAL = {
    'es04feb12_1': 365,
    'es09feb12_2': 365,
    'es09feb12_3': 365,
    'es12mar12_2': 401,
    'es17mar12_1': 401,
    'es17mar12_2': 401,
    'es20may12_1': 371,
    'es21may12_1': 361,
    'es25mar12_6': 401,
    'es25nov11_3': 365,
    'es25nov11_5': 365,
    'es25nov11_9': 365,
    'es25nov11_12': 365,
    'es25nov11_13': 365,
    'es27mar12.012': 399,
    'es27mar12.013': 399,
    'es27mar12_2': 399,
    'es27mar12_3': 399,
    'm258r1_7': 258,
    'm258r1_42': 258,
    'm258r1_44': 258,
    'm258r1_48': 258,
    'm361r2_13': 361,
    'm361r2_17': 361,
    'm361r2_20': 361,
    'm361r2_34': 361,
    'm371r2_3': 371,
    'm531r1_10': 531,
    'm531r1_11': 531,
    'm531r1_29': 531,
    'm531r1_31': 531,
    'm531r1_32': 531,
    'm531r1_34': 531,
    'm531r1_35': 531,
    'm531r1_36': 531,
    'm531r1_38': 531,
    'm531r1_40': 531,
    'm531r1_41': 531,
    'm531r1_42': 531,
    'm531r1_43': 531,
    'm649r1_3': 649,
    'm649r1_5': 649,
    'm649r1_14': 649,
    'm649r1_16': 649,
    'm649r1_17': 649,
    'm649r1_19': 649,
    'm649r1_21': 649,
    'm649r1_22': 649
}

feature_names = ['dep_red',	'dep_sd', 'graph_avg_speed', 'graph_slowest_path', 'graph_fastest_path',
                 'geometrical_avg_shift', 'geometrical_shift_sd', 'geometrical_max_dist', 'spatial_dispersion_count',
                 'spatial_dispersion_sd', 'da', 'da_sd', 'Channels contrast', 'break_measure', 'fwhm', 'get_acc',
                 'max_speed', 'peak2peak', 'trough2peak', 'rise_coef', 'smile_cry',	't_time', 'd_kl', 'jump',
                 'psd_center', 'der_psd_center', 'rise_time', 'unif_dist']

