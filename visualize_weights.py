import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os


def make_word_weights_plot(word_df, base_save_path):
    x = word_df.index.values
    y = word_df.weights.values
    x = np.flipud(x)
    y = np.flipud(y)

    cmap = plt.get_cmap('RdBu_r')
    normalized_data = (y - np.min(y)) / (np.max(y) - np.min(y))
    colors = cmap(normalized_data)

    sns.set_theme()
    sns.set_style("ticks")

    plt.xticks(rotation=90)
    norm = plt.Normalize(vmin=np.min(y), vmax=np.max(y))
    plt.bar(x, y, color=colors, edgecolor="black")
    ax = plt.gca()

    ax.set_xticks(np.arange(len(x)))
    ax.set_xticklabels(x, ha="right", rotation=60, rotation_mode='anchor', fontweight="bold")
    yticks = ax.get_yticks()
    ax.set_yticks(yticks)
    ax.set_yticklabels(np.round(yticks, 2), fontweight="bold")

    plt.ylabel("Malignancy Weight", fontweight="bold", size=14)
    sns.despine(top=True, right=True)
    plt.tight_layout()

    for ext in ['png', 'pdf']:
        plt.savefig(f'{base_save_path}.{ext}', dpi=300, bbox_inches='tight', pad_inches=0.05)

