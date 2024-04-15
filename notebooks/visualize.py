import os
import json

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.ticker as ticker


def draw_confusion_matrices(data, c12n_category, label_type, dataset_dir_path, set_name):

    if not dataset_dir_path.endswith('/'):
        dataset_dir_path += '/'

    categories = list(data['category_to_prediction_stats'].keys())
    categories.sort()

    n = len(categories)

    # Calculate the number of rows and columns for the subplots
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))

    # Create a figure and axes
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*5, nrows*4))

    # Flatten the axes array and iterate over it and the categories simultaneously
    for ax, category in zip(axs.flatten(), categories):
        stats = data['category_to_prediction_stats'][category]

        # Create a confusion matrix
        confusion_matrix = np.array([[stats['true-positive'], stats['false-negative']],
                                     [stats['false-positive'], stats['true-negative']]])

        # Calculate precision, recall, and F1 score
        tp, fp, fn, tn = confusion_matrix.flatten()
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

        # Create a heatmap
        sns.heatmap(confusion_matrix, annot=True, fmt='.0f', cmap='Blues',
                    xticklabels=['Predicted Positive', 'Predicted Negative'],
                    yticklabels=['Actual Positive', 'Actual Negative'], ax=ax)

        # Set the title and add precision, recall, and F1 score
        ax.set_title(f'{category}\n\nPrecision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1_score:.2f}')


    # Remove unused subplots
    for ax in axs.flatten()[n:]:
        fig.delaxes(ax)

    # Set the title
    fig.suptitle(f'Label Type: {label_type} | Category: {c12n_category}', fontsize=16, y=0.98)

    # Set the layout
    plt.tight_layout(pad=2.0)

    # Adjust the space between subplots
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    # Show the plot
    plt.savefig(f'{dataset_dir_path}inference-stats-{set_name}.png')

    print('Saved figure: ' + f'{dataset_dir_path}inference-stats-{set_name}.png')


def visualize_results(label_type, c12n_category, all_stats, dataset_dir_path, inference_set_dir):
    # create a figure and axis
    fig, ax = plt.subplots()

    # set the title
    ax.set_title(f'Label Type: {label_type} | Category: {c12n_category}')

    fig.savefig(f'{dataset_dir_path}{label_type}-{c12n_category}.png')

    # get the categories from all stats
    return


def read_json_file(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data


if __name__ == '__main__':
    dataset_dir_path = '../datasets/crops-surfaceproblem-tags/'
    set_name = 'test'
    inference_file_path = f'{dataset_dir_path}inference-stats-{set_name}.json'
    all_stats = read_json_file(inference_file_path)
    # dataset_dir_path = '../datasets/crops-obstacle-tags/'
    # inference_set_dir = '../datasets/crops-obstacle-tags/test/'
    # visualize_results('surfaceproblem', 'crops', all_stats, dataset_dir_path, inference_set_dir)
    c12n_category = 'tags'
    label_type = 'surfaceproblem'

    draw_confusion_matrices(all_stats, c12n_category, label_type, dataset_dir_path, set_name)
