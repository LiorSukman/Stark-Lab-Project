import numpy as np

def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a, idx, axis=axis)

def calc_perm_test(x, y, reps, statistic=lambda a, b, axis: np.median(a, axis=axis) - np.median(b, axis=axis)):
    n1, n2 = len(x), len(y)
    stat_val = statistic(x, y, axis=0)  # (np.median(x) - np.median(y)) / (np.median(x) +np.median(y))
    xy_comb = np.concatenate((x, y))

    mixed_inds = shuffle_along_axis(np.ones((reps, n1+n2)) * np.arange(n1+n2), axis=1).astype(np.int32)
    all_mixed = xy_comb[mixed_inds]
    mixed_x, mixed_y = all_mixed[:, :n1], all_mixed[:, n1:]
    mixed_stat_val = statistic(mixed_x, mixed_y, axis=1)  # (np.median(mixed_x, axis=1) - np.median(mixed_y, axis=1)) / np.median(mixed_x, axis=1) + np.median(mixed_y, axis=1)
    if stat_val < 0:
        p_val = (np.sum(stat_val >= mixed_stat_val) + 1) / (reps + 1)
        return p_val
    elif stat_val > 0:
        p_val = (np.sum(stat_val <= mixed_stat_val) + 1) / (reps + 1)
        return p_val
    else:
        return 1


