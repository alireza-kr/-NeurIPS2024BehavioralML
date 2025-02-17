import numpy as np
from scipy.io import loadmat
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

from utils import upper_tri


def plot_res(data, model_name):
    # Create a figure and subplots (3 subplots horizontally)
    fig, axes = plt.subplots(1, 3, figsize=(18, 3), sharey=True)

    # Colors for the different points
    colors = ['red', 'blue', 'green']
    markers = ['o', 's', '^']  # 'o' = circle, 's' = square, '^' = triangle
    subplot_names = [f'{model_name} ImageNet', f'{model_name} DeWind', f'{model_name} Untrained']
    labels = ['Full', 'Prune', 'ANOVA']

    # Loop through each 2D slice of the matrix
    for k in range(data.shape[0]):
        ax = axes[k]  # Select the current subplot

        # Loop through each row in the 2D slice
        for i in range(data.shape[1]):
            # Plot each value in the row as a point on the horizontal bar
            for j in range(data.shape[2]):
                ax.scatter(data[k, i, j], i, color=colors[j], s=100, marker=markers[j])  # s is the size of the points

        # Set y-axis labels (one label per row)
        ax.set_yticks(range(data.shape[1]))
        ax.set_yticklabels(['V1', 'V2', 'V4', 'IT'])
        ax.tick_params(axis='both', which='major', labelsize=20)

        # Set x-axis limits
        ax.set_xlim(0, 1)

        # Remove upper and lower spines
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # Add grid with dashed style on y-axis
        ax.grid(True, axis='x', linestyle='--')
        ax.grid(True, axis='y', linestyle='-', linewidth=2)
        # Add title for each subplot
        ax.set_title(subplot_names[k], fontsize=22)
        # Add legend only for the first subplot
        # Add legend only for the first subplot, with larger text size
        if k == 1:
            handles = [plt.Line2D([0], [0], color=colors[j], marker=markers[j], lw=0, markersize=10) for j in
                       range(len(labels))]
            ax.legend(handles, labels, loc='upper right', fontsize=15)

    # Set common labels
    # fig.suptitle('CORnet-Z')
    # fig.text(0.5, 0.04, 'Pearson correlation', ha='center', fontsize=12)
    # fig.text(0.04, 0.5, 'Bars', va='center', rotation='vertical', fontsize=12)

    # Reduce the gap between y-axis labels and the plot
    plt.subplots_adjust(left=0.1, right=0.9, wspace=0.3)

    # Adjust layout to be tight
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig(f'./res/figures/{model_name}_rsa.png')


def read_selection_res(model = 'CORnet-Z', dataset = 'ImageNet'):
    print(f'Model {model}, trained on {dataset} dataset')

    # read the result
    network_layers = ['V1', 'V2', 'V4', 'IT']
    # brain_area = 'IPS345'
    # rdm_human = loadmat('./data/MRI-RDM.mat', simplify_cells=True)['RDM'][brain_area]
    human_data = 'behavior'
    rdm_human = loadmat('./data/Number.mat', simplify_cells=True)['Number']
    rdm_human_trim = upper_tri(rdm_human)
    rdm_glove = np.load('./data/rdm_glove.npy')
    rdm_glove_trim = upper_tri(rdm_glove)

    for layer in network_layers:
        # Load pruning results (score Pruning with Human RDM)
        print(f'\n--- Layer {layer} ---')
        res = loadmat(f'./res/selection/forward/{human_data}/{model}/{dataset}/{layer}.mat')
        score_deviation = res['score_full'] - res['score_each_node']
        rank_deviation = np.argsort(score_deviation[0])[::-1] # sort from highest to lowest
        max_position = np.argmax(res['score_sfs'])
        selected_nodes = rank_deviation[:max_position+1]
        print('[Pruning] Full features - RSA Full - Retained features - RSA Retained')
        print(len(rank_deviation), round(res['score_full'][0][0], 2),
              len(selected_nodes), round(np.max(res['score_sfs']), 2))

        # Load full activations
        acts = np.load(f'./data/{model}/{dataset}/{layer}.npy')
        acts_avg = np.array([np.mean(acts[i:i + 100], axis=0) for i in range(0, 3200, 100)])
        rdm_acts = 1 - np.array(np.corrcoef(acts_avg))
        rdm_acts_trim = upper_tri(rdm_acts)

        # Score Pruning with Glove RDM
        score_glove_full = pearsonr(rdm_glove_trim, rdm_acts_trim)[0]
        acts_pruning = acts_avg[:, selected_nodes]
        rdm_acts_pruning = 1 - np.array(np.corrcoef(acts_pruning))
        rdm_acts_trim_pruning = upper_tri(rdm_acts_pruning)
        score_glove_pruning = pearsonr(rdm_glove_trim, rdm_acts_trim_pruning)[0]
        print('[Pruning] RSA Full Glove - RSA Retained Glove')
        print(round(score_glove_full, 2), round(score_glove_pruning, 2))

        # Load ANOVA units
        num_unit = np.load(f'./res/num_unit/{model}/{dataset}/{layer}.npy')
        if len(num_unit) > 0:
            overlap = len(set(selected_nodes).intersection(set(num_unit))) / min(len(selected_nodes), len(num_unit))
        else:
            overlap = np.nan

        if len(num_unit) > 1:
            acts_anova = acts_avg[:, num_unit]
            rdm_acts_anova = 1 - np.array(np.corrcoef(acts_anova))
            rdm_acts_anova_trim = upper_tri(rdm_acts_anova)
            score_anova_human = pearsonr(rdm_human_trim, rdm_acts_anova_trim)[0]
            score_anova_glove = pearsonr(rdm_glove_trim, rdm_acts_anova_trim)[0]
        else:
            score_anova_human = np.nan
            score_anova_glove = np.nan
        print('[ANOVA] ANOVA features - RSA ANOVA Human RDM - RSA ANOVA Glove RDM - Overlap Pruning-ANOVA')
        print(len(num_unit), round(score_anova_human, 2), round(score_anova_glove, 2), round(overlap, 2))



if __name__ == '__main__':
    # cornet_z_data = np.array([[[0.38, 0.58, 0.48], [0.43, 0.61, 0.58], [0.46, 0.64, 0.26], [0.32, 0.66, 0.27]],
    #                           [[0.38, 0.57, 0.48], [0.43, 0.6, 0.5], [0.45, 0.62, 0.28], [0.56, 0.68, 0.24]],
    #                           [[0.28, 0.5, 0.19], [0.33, 0.52, np.nan], [0.32, 0.53, 0.11], [0.31, 0.5,0.01]]])
    # plot_res(data=cornet_z_data, model_name='CORnet-Z')

    # cornet_s_data = np.array([[[0.24, 0.71, 0.56], [0.24, 0.78, 0.74], [0.37, 0.82, 0.79], [0.29, 0.84, 0.72]],
    #                           [[0.42, 0.59, 0.71], [0.35, 0.51, 0.63], [0.26, 0.45, 0.44], [0.36, 0.68, np.nan]],
    #                           [[0.21, 0.63, 0.01], [0.11, 0.68, 0.41], [0.03, 0.73, 0.45], [0.03, 0.75, 0.7]]])
    # plot_res(data=cornet_s_data, model_name='CORnet-S')

    #
    model = 'CORnet-Z' # CORnet-Z CORnet-S
    dataset = 'ImageNet' # ImageNet DeWind Untrained
    read_selection_res(model=model, dataset=dataset)
