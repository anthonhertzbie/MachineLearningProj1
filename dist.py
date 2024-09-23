import numpy as np
import matplotlib.pyplot as plt

def histograms(normalized_df, attribute_labels):
    num_cols = normalized_df.shape[1]
    for col in range(num_cols):
        col_data = normalized_df[:, col]

        # Create histogram
        plt.hist(col_data, bins=5, edgecolor='black') 
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of Attribute {attribute_labels[col]}')
        plt.show()
