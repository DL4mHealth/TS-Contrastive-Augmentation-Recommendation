import numpy as np
from scipy import signal
from sklearn.preprocessing import StandardScaler
import math
pi = math.pi

scaler = StandardScaler()


def synth_trend_season(query_period, query_length):
    n_sample = 1
    '''method 2: only take one period of synthetic season, so len_sample=real_stl_period'''
    real_stl_period = query_period
    len_sample = real_stl_period
    n_period = len_sample/real_stl_period
    t = np.linspace(0, 10, len_sample)  # time range
    t.shape = [1, len_sample]
    t_matrix = np.repeat(t, n_sample, axis=0)

    # Trends should not be compared segment by segment
    real_length = query_length
    trend_length = real_length

    trend_t = np.linspace(0, 10, trend_length)  # time range
    trend_t.shape = [1, trend_length]
    trend_t_matrix = np.repeat(trend_t, n_sample, axis=0)

    '''trend_co: 0.2'''
    trend_co = 0.2
    T1_p = trend_co*trend_t_matrix
    T1_n = -trend_co*trend_t_matrix

    T2_p = (trend_t_matrix + 0.1)**trend_co
    T2_n = (trend_t_matrix + 0.1)**(-trend_co)

    T1_all = np.vstack((T1_p, T1_n))
    T1_all = scaler.fit_transform(T1_all.T)
    T1_all = T1_all.T

    T2_all = np.vstack((T2_p, T2_n))
    T2_all = scaler.fit_transform(T2_all.T)
    T2_all = T2_all.T

    '''Trend 1, 2'''
    '''Season_co: 1, 5, 10 -> 1, 3, 5 -> 1, 2, 3 -> 1'''
    s_co = n_period  # make no change when using S1

    s1_w = [0.1, 0.5, 0.9]
    S1_all = np.zeros([len(s1_w), len_sample])
    s1_r = 0

    for s1w in s1_w:
        S1 = s1w * np.sin(s_co * t_matrix) + (1 - s1w) * np.cos(s_co * t_matrix)
        S1_all[s1_r, :] = S1
        s1_r = s1_r+1

    '''normalize'''
    S1_all = scaler.fit_transform(S1_all.T)
    S1_all = S1_all.T

    # Season 2
    s2_w = [4, 5, 6]
    S2_all = np.zeros([len(s1_w), len_sample])
    l = int(real_stl_period)

    s2_r = 0
    for s2w in s2_w:
        wavelet = signal.morlet(l, w=s2w, s=1)
        wavelet = wavelet.real
        S2_long = np.tile(wavelet, int(len_sample/l))
        S2_all[s2_r, :] = S2_long
        s2_r = s2_r+1

    '''normalization'''
    S2_all = scaler.fit_transform(S2_all.T)
    S2_all = S2_all.T

    return T1_all, T2_all, S1_all, S2_all


