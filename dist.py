import numpy as np
import matplotlib.pyplot as plt

def histograms(normalized_df, attribute_labels):
    num_cols = normalized_df.shape[1]

    _, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()

    for col in range(num_cols):
        col_data = normalized_df[:, col]

        ax = axes[col]

        ax.hist(col_data, bins=8, edgecolor='black') 
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Histogram of Attribute {attribute_labels[col]}')
        
    plt.tight_layout(pad=2.0)
    plt.show()