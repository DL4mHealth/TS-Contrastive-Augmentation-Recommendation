"""
Author: Ziyu Liu (ziyu.liu2@student.rmit.edu.au)
Code for Paper: Guidelines for Augmentation Selection in Contrastive Learning for Time Series Classification
July 07 2024
"""

from statsmodels.tsa.seasonal import STL
import numpy as np
import pickle
from tqdm import tqdm
from .utlis import Average, zscore_norm, select_group, power_weight, recommend, popularity_recommend
from sklearn.metrics.pairwise import cosine_similarity
from .synth_trend_season import synth_trend_season
import os


def aug_rec_ts(queryset_name='QueryTS', K=3, query_length=1280, query_period_list=[40], queryset=None):
    """
    :param queryset_name: string, name of your query dataset. Here we take 'QueryTS' as default
    :param K: int, the number of Augmentations you want to have
    :param query_length: int, feature length in query dataset
    :param query_period_list: list, a list of potential period (length of a single season) such as [1280, 640, 320, 160, 80, 40].
    Or, you can specify a single period like [40]. query_length should be divisible by each element in period list
    :param queryset: numpy array, the training features of your query dataset, 2D [#-samples, #-features]
    :return: list of string, Top K effective contrastive augmentations for your query time series dataset
    """

    '''Step 1: Decomposition the query dataset'''
    for query_period in query_period_list:
        queryset = zscore_norm(queryset)
        reshaped_data = queryset.reshape(-1, queryset.shape[2])  # Reshape the dataset to make one time series per row
        query_trend = np.zeros_like(reshaped_data)
        query_season = np.zeros_like(reshaped_data)
        for i in tqdm(range(reshaped_data.shape[0]), desc="Processing"):
            time_series = reshaped_data[i, :]
            stl_result = STL(time_series, period=query_period).fit()
            query_trend[i, :] = stl_result.trend
            query_season[i, :] = stl_result.seasonal
        # print(query_trend.shape, query_season.shape)
        folder_path = 'Queryset_related/' + queryset_name + "_decomp/"
        realtrend_path = queryset_name + "_trend_" + str(query_period) + ".pkl"
        realseason_path = queryset_name + "_season_" + str(query_period) + ".pkl"

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        pickle.dump(query_trend, open(folder_path + realtrend_path, 'wb'))
        pickle.dump(query_season, open(folder_path + realseason_path, 'wb'))


    '''Step 2 & 3: calculate similarity'''
    T1_sim_list, T2_sim_list, S1_sim_list, S2_sim_list = [], [], [], []
    power_trend_list, power_season_list = [], []
    for query_period in query_period_list:
        folder_path = 'Queryset_related/' + queryset_name + "_decomp/"
        realtrend_path = queryset_name + "_trend_" + str(query_period) + ".pkl"
        realseason_path = queryset_name + "_season_" + str(query_period) + ".pkl"
        query_trend = pickle.load(open(folder_path + realtrend_path, "rb"))
        query_season = pickle.load(open(folder_path + realseason_path, "rb"))

        T1, T2, S1, S2 = synth_trend_season(query_period, query_length)
        T1_sim = np.mean(np.abs(cosine_similarity(T1, query_trend)))
        T2_sim = np.mean(np.abs(cosine_similarity(T2, query_trend)))
        """Period Average: New version"""
        query_stacked = query_season.reshape(-1, query_period)
        S1_sim = np.mean(np.abs(cosine_similarity(S1, query_stacked)))
        S2_sim = np.mean(np.abs(cosine_similarity(S2, query_stacked)))

        T1_sim_list.append(T1_sim)
        T2_sim_list.append(T2_sim)
        S1_sim_list.append(S1_sim)
        S2_sim_list.append(S2_sim)
        # print('Period and similarities:', query_period, T1_sim, T2_sim, S1_sim, S2_sim)

        """Power calculation"""
        power_trend = np.mean(query_trend ** 2)
        power_season = np.mean(query_season ** 2)
        power_trend_list.append(power_trend)
        power_season_list.append(power_season)

    T1_sim, T2_sim, S1_sim, S2_sim = Average(T1_sim_list), Average(T2_sim_list), Average(S1_sim_list), Average(S2_sim_list)
    power_trend, power_season = Average(power_trend_list), Average(power_season_list)

    print(f"T1 Similarity: {T1_sim:.4f}\nT2 Similarity: {T2_sim:.4f}\n"
          f"S1 Similarity: {S1_sim:.4f}\nS2 Similarity: {S2_sim:.4f}\n")

    '''Step 4: select group'''
    twin_group = select_group(T1_sim, T2_sim, S1_sim, S2_sim)
    if twin_group == None:
        print("The trend and season of the query dataset are not similar to any synthetic trend and season.\n"
              "Stay tuned for our next version.")
        exit(0)

    '''Step 5: calculate power'''
    print(f"Trend Power:{power_trend}\nSeason Power:{power_season}")

    '''Step 6: select twin dataset and make recommendataion'''
    twin_suffix = power_weight(power_trend/power_season)
    top_augmentation = recommend(twin_group, twin_suffix)
    print('Your twin dataset is:', twin_group, twin_suffix)
    print(f"\033[1;32mTrend-season based top {K} augmentations are: {top_augmentation[:K]}\033[0m")
    return top_augmentation[:K]


def aug_rec_popular(K=3):  # K<=9
    """If you want to use the popularity-based recommendation:"""
    poprank = popularity_recommend()
    print(f'Popularity-based top {K} recommendation: {poprank[:K]}')
    return poprank[:K]

