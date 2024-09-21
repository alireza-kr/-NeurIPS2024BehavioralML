import numpy as np

def upper_tri(r):
    # author: priya tarigopula
    # Extract off-diagonal elements of each Matrix
    ioffdiag = np.triu_indices(r.shape[0], k=1)  # indices of off-diagonal elements
    r_offdiag = r[ioffdiag]
    return r_offdiag


# def select_forward_old(rdm_human, acts, rank_node):
#     """
#     Sequentially add each node to the embedding, then score RSA against human RDM.
#
#     This function iteratively selects the top `i` nodes (columns) from the activation matrix `acts` based on
#     the given `rank_node`, computes the Representational Dissimilarity Matrix (RDM) for the selected nodes,
#     and calculates the RSA score with the human RDM.
#
#     :param rdm_human: (numpy.ndarray): The human Representational Dissimilarity Matrix (RDM).
#     :param acts: (numpy.ndarray): The activation matrix, where each column represents a node.
#     :param rank_node: (numpy.ndarray): The ranking of nodes, where each element represents the index
#                                            of a node in order of selection.
#     :return: numpy.ndarray: An array of RSA scores, where each score corresponds to the RSA for the selected
#                        top `i` nodes.
#     """
#     rdm_human_trim = upper_tri(rdm_human)
#     rsa_scores = []
#     for i in tqdm(range(2, acts.shape[1]+1)):
#         rdm_acts = 1 - np.corrcoef(acts[:, rank_node[:i]]) # select top i columns, then compute the RDM
#         rsa_scores.append(pearsonr(rdm_human_trim, upper_tri(rdm_acts))[0])
#     return np.array(rsa_scores)
#
#
# def remove_node_n_eval_old(rdm_human, acts):
#     """
#     Evaluate the impact of removing each node on Representational Similarity Analysis (RSA) scores.
#
#     This function iteratively removes each node (column) from the activation matrix `acts`,
#     computes the Representational Dissimilarity Matrix (RDM) for the modified matrix, and
#     calculates the RSA score with the human RDM. The RSA score is computed as the Pearson
#     correlation between the upper triangular parts of the human RDM and the modified RDM.
#
#     :param rdm_human: (numpy.ndarray): The human Representational Dissimilarity Matrix (RDM).
#     :param acts: (numpy.ndarray): The activation matrix, where each column represents a node.
#     :return: numpy.ndarray: An array of RSA scores, where each score corresponds to the RSA after
#                        removing the respective node.
#     """
#     rdm_human_trim = upper_tri(rdm_human)
#     rsa_scores = []
#     for i in tqdm(range(acts.shape[1])):
#         rdm_acts = 1 - np.corrcoef(np.delete(acts, i, axis=1)) # remove i column, then compute the RDM
#         rsa_scores.append(pearsonr(rdm_human_trim, upper_tri(rdm_acts))[0])
#     return np.array(rsa_scores)
