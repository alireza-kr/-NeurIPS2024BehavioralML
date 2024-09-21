import numpy as np
from scipy.io import loadmat, savemat
from scipy.stats import pearsonr
from tqdm import tqdm
import joblib
import multiprocessing

from utils import upper_tri


def remove_node_n_eval(rdm_human, acts):
    """
    Evaluate the impact of removing each node on Representational Similarity Analysis (RSA) scores.

    This function iteratively removes each node (column) from the activation matrix `acts`,
    computes the Representational Dissimilarity Matrix (RDM) for the modified matrix, and
    calculates the RSA score with the human RDM. The RSA score is computed as the Pearson
    correlation between the upper triangular parts of the human RDM and the modified RDM.

    :param rdm_human: (numpy.ndarray): The human Representational Dissimilarity Matrix (RDM).
    :param acts: (numpy.ndarray): The activation matrix, where each column represents a node.
    :return: numpy.ndarray: An array of RSA scores, where each score corresponds to the RSA after
                       removing the respective node.
    """
    rdm_human_trim = upper_tri(rdm_human)
    def _repeat_func(node_idx):
        rdm_acts = 1 - np.corrcoef(np.delete(acts, node_idx, axis=1))  # remove i column, then compute the RDM
        return pearsonr(rdm_human_trim, upper_tri(rdm_acts))[0]

    rsa_scores = joblib.Parallel(n_jobs=NUM_CPU_USE, prefer='processes')(joblib.delayed(_repeat_func)(i)
                                                                       for i in tqdm(range(acts.shape[1])))
    return np.array(rsa_scores)


def select_forward(rdm_human, acts, rank_node):
    """
    Sequentially add each node to the embedding, then score RSA against human RDM.

    This function iteratively selects the top `i` nodes (columns) from the activation matrix `acts` based on
    the given `rank_node`, computes the Representational Dissimilarity Matrix (RDM) for the selected nodes,
    and calculates the RSA score with the human RDM.

    :param rdm_human: (numpy.ndarray): The human Representational Dissimilarity Matrix (RDM).
    :param acts: (numpy.ndarray): The activation matrix, where each column represents a node.
    :param rank_node: (numpy.ndarray): The ranking of nodes, where each element represents the index
                                           of a node in order of selection.
    :return: numpy.ndarray: An array of RSA scores, where each score corresponds to the RSA for the selected
                       top `i` nodes.
    """
    rdm_human_trim = upper_tri(rdm_human)
    def _repeat_func(node_idx):
        rdm_acts = 1 - np.corrcoef(acts[:, rank_node[:node_idx]])  # select top i columns, then compute the RDM
        return pearsonr(rdm_human_trim, upper_tri(rdm_acts))[0]

    rsa_scores = joblib.Parallel(n_jobs=NUM_CPU_USE, prefer='processes')(joblib.delayed(_repeat_func)(i)
                                                                       for i in tqdm(range(2, acts.shape[1]+1)))
    return np.array(rsa_scores)


if __name__ == '__main__':
    # brain_area = 'VO2'
    model = 'CORnet-S'
    dataset = 'Untrained'

    NUM_CPU_USE = int(multiprocessing.cpu_count() * 0.5)
    # rdm_mri = loadmat('./data/MRI-RDM.mat', simplify_cells=True)['RDM'][brain_area]
    rdm_behavior = loadmat('./data/Number.mat', simplify_cells=True)['Number']

    network_layers = ['IT', 'V4', 'V2', 'V1']
    for layer in network_layers:
        print(layer)
        acts = np.load(f'./data/{model}/{dataset}/{layer}.npy')
        # compute the average activations over 100 datapoints
        acts_avg = np.array([np.mean(acts[i:i + 100], axis=0) for i in range(0, 3200, 100)])

        # evaluate the importance of each node
        score_each_node = remove_node_n_eval(rdm_human=rdm_behavior, acts=acts_avg)

        rdm_acts_full = 1 - np.corrcoef(acts_avg) # rdm of the full activations
        score_full = pearsonr(upper_tri(rdm_behavior), upper_tri(rdm_acts_full))[0]

        # compute the deviation when remove each node
        # an important node when removed will largely decrease the original RSA
        score_deviation = score_full - score_each_node
        # score_deviation = score_full ** 2 - score_each_node ** 2

        # rank from highest to lowest
        rank_deviation = np.argsort(score_deviation)[::-1]

        # sequentially select nodes and compute the RSA
        score_forward = select_forward(rdm_human=rdm_behavior, acts=acts_avg, rank_node=rank_deviation)

        res = {'score_full': np.array([score_full]),
               'score_each_node': score_each_node,
               'score_sfs': score_forward,
              }
        # savemat(f'./res/selection/forward/{brain_area}_{layer}.mat', res)
        savemat(f'./res/selection/forward/behavior/{model}/{dataset}/{layer}.mat', res)

