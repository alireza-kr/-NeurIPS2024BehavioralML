import numpy as np
import scipy.stats as stats
from tqdm import tqdm


def perform_three_way_anova(data):
    """
    The goal was to identify units that had a significant
    change in response across different numerosities while maintaining an invariant response across
    variations in total field area and average item area, as well as across the three interactions among
    pairs of factors, and the interaction among all of the three.
    """
    p_values = []
    for unit_data in tqdm(data):
        # Index of conditions
        numerosity = np.repeat(np.arange(n_num), n_tfa * n_aia * n_instances)
        aia = np.tile(np.repeat(np.arange(n_aia), n_tfa * n_instances), 4)
        tfa = np.tile(np.repeat(np.arange(n_tfa), n_instances), 16)

        # Three-way ANOVA
        f_num, p_num = stats.f_oneway(*[unit_data[numerosity == i] for i in range(n_num)])
        f_aia, p_aia = stats.f_oneway(*[unit_data[aia == i] for i in range(n_aia)])
        f_tfa, p_tfa = stats.f_oneway(*[unit_data[tfa == i] for i in range(n_tfa)])
        f_num_aia, p_num_aia = stats.f_oneway(
            *[unit_data[(numerosity == i) & (aia == j)] for i in range(n_num) for j in range(n_aia)])
        f_num_tfa, p_num_tfa = stats.f_oneway(
            *[unit_data[(numerosity == i) & (tfa == j)] for i in range(n_num) for j in range(n_tfa)])
        f_aia_tfa, p_aia_tfa = stats.f_oneway(
            *[unit_data[(aia == i) & (tfa == j)] for i in range(n_aia) for j in range(n_tfa)])
        f_num_aia_tfa, p_num_aia_tfa = stats.f_oneway(
            *[unit_data[(numerosity == i) & (aia == j) & (tfa == k)]
            for i in range(n_num) for j in range(n_aia) for k in range(n_tfa)])

        p_values.append((p_num, p_aia, p_tfa, p_num_aia, p_num_tfa, p_aia_tfa, p_num_aia_tfa))
    return np.array(p_values)


if __name__ == '__main__':
    n_num = 4  # number of numerosities
    n_tfa = 2  # number of total field area groups
    n_aia = 4  # number of average item area groups
    n_instances = 100  # number of instances per numerosity

    model = 'CORnet-Z'
    dataset = 'Untrained'
    for layer in ['IT', 'V1', 'V2', 'V4']:
        network_responses = np.load(f'./data/{model}/{dataset}/{layer}.npy').T # transpose
        p_values = perform_three_way_anova(network_responses)

        # # Print p-values for debugging
        # print("P-values from ANOVA:")
        # print(p_values)

        # Select numerosity-selective units
        alpha = 0.01
        numerosity_selective_units = np.where((p_values[:, 0] < alpha) &
                                              (p_values[:, 1] >= alpha) & (p_values[:, 2] >= alpha) &
                                              (p_values[:, 3] >= alpha) & (p_values[:, 4] >= alpha) &
                                              (p_values[:, 5] >= alpha) & (p_values[:, 6] >= alpha))[0]

        # Print selected numerosity-selective units for debugging
        print(f"Numerosity-selective units (count: {len(numerosity_selective_units)}):")
        print(numerosity_selective_units)
        if len(numerosity_selective_units) == 0:
            print("No numerosity-selective units found. Check your data and ANOVA parameters.")
        np.save(f'./res/num_unit/{model}/{dataset}/{layer}.npy', numerosity_selective_units)
