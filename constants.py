import numpy as np

NUM_CHANNELS = 8
TIMESTEPS = 32
SAMPLE_RATE = 20_000
UPSAMPLE = 8
RICH_FACTOR = 6

# be careful and assert it is not being changed
COORDINATES = np.array([[0, 0], [-9, 20], [8, 40], [-13, 60], [12, 80], [-17, 100], [16, 120], [-21, 140]])
MIN_TIME_LIST = 8000
ACH_WINDOW = 1000
VERBOS = False
DEBUG = False
SEED = 2
INF = 9999

SPATIAL = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, -1]
SPATIAL_R = np.arange(18 * RICH_FACTOR + 1)
SPATIAL_R[-1] = -1
MORPHOLOGICAL = [18, 19, 20, 21, 22, 23, 24, 25, -1]
MORPHOLOGICAL_R = np.arange(8 * RICH_FACTOR + 1) + 18 * RICH_FACTOR
MORPHOLOGICAL_R[-1] = -1
TEMPORAL = [26, 27, 28, 29, 30, 31, 32, 33, -1]
TEMPORAL_R = np.arange(8 * RICH_FACTOR + 1) + (18 + 8) * RICH_FACTOR
TEMPORAL_R[-1] = -1

# use the following if adding t-time
# TRANS_MORPH = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, -1]
TRANS_MORPH = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, -1]

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
feature_names_org = ['spatial_dispersion_count', 'spatial_dispersion_sd', 'spatial_dispersion_area', 'dep_red',
                     'dep_sd', 'fzc_red', 'fzc_sd', 'szc_red', 'szc_sd', 'dep_graph_avg_speed',
                     'dep_graph_slowest_path', 'dep_graph_fastest_path', 'fzc_graph_avg_speed',
                     'fzc_graph_slowest_path', 'fzc_graph_fastest_path', 'szc_graph_avg_speed',
                     'szc_graph_slowest_path', 'szc_graph_fastest_path', 'break_measure', 'fwhm', 'get_acc',
                     'max_speed', 'peak2peak', 'trough2peak', 'rise_coef', 'smile_cry', 'firing_rate', 'd_kl_start',
                     'd_kl_mid', 'jump', 'psd_center', 'der_psd_center', 'rise_time', 'unif_dist']

feature_names_rich = []
for f in feature_names_org:
    feature_names_rich += [f'{f}', f'{f}_avg', f'{f}_std', f'{f}_q25', f'{f}_q50', f'{f}_q75']
