import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata
import pkg_resources

var_list = ['a1', 'a2', 'a3', 'b1', 'b2', 'b3', 'c1', 'c2', 'c3', 'd1', 'd2', 'd3',
            'HAR', 'PTB', 'FD', 'MP', 'ElecD', 'SPX']
augs_list = ["Jittering",  "Scaling",  "Flipping",  "Permutation",  "Resizing",  "TimeMasking",
             "Freq_RandomMasking",  "Timewise_neighboring",  'NoPreTrain']

synthetic_f1 = np.load(pkg_resources.resource_filename(__name__, 'synthetic_F1_array.npy'))


def Average(lst):
    return sum(lst) / len(lst)

def recall_at_k(actual_list, predicted_list, k):
    """
    Calculate Recall@K metric.

    Parameters:
    actual_list (list): List of actual relevant items.
    predicted_list (list): List of predicted items.
    k (int): Top K items to consider.

    Returns:
    recall (float): Recall@K score.
    """
    # Calculate the number of relevant items in actual_list
    num_relevant = sum(1 for item in actual_list[:k] if item in predicted_list[:k])

    # Calculate total number of relevant items in actual_list
    total_relevant = k #len(actual_list)

    # Calculate Recall@K
    recall = num_relevant / total_relevant if total_relevant > 0 else 0

    return recall


def rank_value(rank_realvalue, margin=0.01):
    margin = margin*rank_realvalue[-1]
    # margin is a ratio for threshold, rank_realvalue[-1] is the no-pretrain performance
    # Create a dictionary to store counts of smaller elements
    rank_dict = {}
    for i, value in enumerate(rank_realvalue):
        count = 0
        for other_value in rank_realvalue:
            if other_value > value+margin:
                count += 1
        rank_dict[value] = count + 1  # Adding 1 to start ranking from 1
    return list(rank_dict.values())


def print_importance_ranked_aug(importance_rankings):
    augs_list = ["Jittering", "Scaling", "Flipping", "Permutation", "Resizing", "TimeMasking",
             "Freq_RandomMasking", "Timewise_neighboring", 'NoPreTrain']
    # Combine the list and rankings into tuples for sorting
    combined = list(zip(augs_list, importance_rankings))

    # Sort the combined list based on importance rankings, maintaining the original order for ties
    sorted_combined = sorted(combined, key=lambda x: (x[1], combined.index(x)))

    # Extract the sorted elements
    sorted_augs_list = [x[0] for x in sorted_combined]
    return sorted_augs_list


def zscore_norm(x):
    if not isinstance(x, np.ndarray):
        x = x.numpy()
    train_x_norm = x.copy()
    for i in range(x.shape[1]):
        x_channel = x[:, i, :]
        scaler = StandardScaler()
        x_channel_norm = scaler.fit_transform(x_channel)
        train_x_norm[:, i, :] = x_channel_norm

    return train_x_norm


def divergence_score(t1, t2):
    return (max(t1, t2)-min(t1, t2))/((t1+t2)/2)


def select_group(T1_sim, T2_sim, S1_sim, S2_sim, dscore_threshold=0.05):
    T_dscore = divergence_score(T1_sim, T2_sim)
    S_dscore = divergence_score(S1_sim, S2_sim)

    if (T_dscore >= dscore_threshold) & (S_dscore >= dscore_threshold):
        if (T1_sim > T2_sim) & (S1_sim > S2_sim):
            twin_group = "A"
        elif (T1_sim > T2_sim) & (S1_sim < S2_sim):
            twin_group = "B"
        elif (T1_sim < T2_sim) & (S1_sim > S2_sim):
            twin_group = "C"
        elif (T1_sim < T2_sim) & (S1_sim < S2_sim):
            twin_group = "D"
    elif (T_dscore >= dscore_threshold) & (S_dscore < dscore_threshold):
        if T1_sim > T2_sim:
            twin_group = "AB"
        elif T1_sim < T2_sim:
            twin_group = "CD"
    elif (T_dscore < dscore_threshold) & (S_dscore >= dscore_threshold):
        if S1_sim > S2_sim:
            twin_group = "AC"
        elif S1_sim < S2_sim:
            twin_group = "BD"
    else:
        twin_group =None
        # print("The trend and season of the query dataset are not similar to any synthetic trend and season.\n"
        #       "Stay tuned for our next version.")
    # try:
    #     print("The twin group is:", twin_group)
    # except:
    #     pass
    # return twin_group, T_dscore, S_dscore
    return twin_group


def power_weight(c_w):
    if c_w <= 5/9:
        twin_suffix = "1"
    elif 5/9 <= c_w <= 5:
        twin_suffix = "2"
    elif c_w >= 5:
        twin_suffix = "3"

    return twin_suffix


def popularity_recommend():
    '''popularity based ranking'''
    # synthetic_f1 = np.load("synthetic_F1_array.npy")
    pp_rank_matrix = np.zeros((12, 9))
    for i in range(12):
        pp_rank_matrix[i] = np.array(rank_value(synthetic_f1[i]))
    popular_aug_rank = rankdata(list(pp_rank_matrix.sum(axis=0)), 'min')
    popularity_rank = print_importance_ranked_aug(popular_aug_rank)
    # print(popularity_rank)
    return popularity_rank



def recommend(twin_group, twin_suffix):
    # twin_group = 'A'
    rank_matrix = np.zeros((12, 9))
    for i in range(12):
        rank_matrix[i] = np.array(rank_value(synthetic_f1[i], margin=0))

    if len(twin_group) == 1:
        twin_dataset = twin_group + twin_suffix
        # top_augmentation = results_dic[twin_dataset.lower()]
        top_augmentation = print_importance_ranked_aug(rankdata(rank_matrix[var_list.index(twin_dataset.lower())]))

    elif len(twin_group) == 2:
        twin_0 = twin_group[0] + twin_suffix
        twin_1 = twin_group[1] + twin_suffix
        rank_twin_0, rank_twin_1 = rank_matrix[var_list.index(twin_0.lower())], rank_matrix[
            var_list.index(twin_1.lower())]
        top_augmentation = print_importance_ranked_aug(
            rankdata(list(np.vstack((rank_twin_0, rank_twin_1)).sum(axis=0))))
    return top_augmentation

