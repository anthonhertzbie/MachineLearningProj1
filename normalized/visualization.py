import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


def project_to3d(title, normalized_df, sorted_eigenvectors, strength_norm):
    basis_vectors = sorted_eigenvectors[:, 0:3]
    projected_data = np.dot(normalized_df, basis_vectors)
    x = projected_data[:, 0]
    y = projected_data[:, 1]
    z = projected_data[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=strength_norm, cmap='RdBu_r')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_xlim(-350, 220)
    ax.set_ylim(-310, 200)
    ax.set_zlim(-300, 250)
    ax.set_title(title)
    plt.savefig(f'plots/{title}.png')
    plt.show()

    df = pd.DataFrame(projected_data)
    df['normalized_comp_strength'] = strength_norm
    df.sort_values('normalized_comp_strength', ascending=False, inplace=True)
    first_20_percent = int(0.20 * len(df))
    df_first_20_percent = df.iloc[:first_20_percent]

    x = df_first_20_percent[0]
    y = df_first_20_percent[1]
    z = df_first_20_percent[2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='red')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_xlim(-350, 220)
    ax.set_ylim(-310, 200)
    ax.set_zlim(-300, 250)
    plt.title('Plot of Strongest 20% of Data')
    plt.savefig(f'plots/best3d.png')
    plt.show()

    df = pd.DataFrame(projected_data)
    df['normalized_comp_strength'] = strength_norm
    df.sort_values('normalized_comp_strength', ascending=True, inplace=True)
    first_20_percent = int(0.20 * len(df))
    df_first_20_percent = df.iloc[:first_20_percent]
    x = df_first_20_percent[0]
    y = df_first_20_percent[1]
    z = df_first_20_percent[2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='blue')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_xlim(-350, 220)  
    ax.set_ylim(-310, 200)  
    ax.set_zlim(-300, 250)  
    plt.title('Plot of Weakest 20% of Data')
    plt.savefig(f'plots/worst3d.png')
    plt.show()

    return basis_vectors


def project_to2d(title, cbar_label, normalized_df, eigenvectors, strength_norm):
    # Project data to the first two principal components (2D space)
    basis_vectors = eigenvectors[:, 0:2]
    projected_data = np.dot(normalized_df, basis_vectors)
    x = projected_data[:, 0]
    y = projected_data[:, 1]

    # Plot the full scatter plot with color based on strength_norm
    plt.scatter(x, y, c=strength_norm, cmap='RdBu_r')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.xlim(-350, 220)
    plt.ylim(-310, 200)
    plt.title(title)
    cbar = plt.colorbar()
    cbar.set_label(cbar_label)
    plt.savefig(f'plots/{title}.png')
    plt.show()

    # Create DataFrame from the projected data for easier manipulation
    df = pd.DataFrame(projected_data)
    df['normalized_comp_strength'] = strength_norm

    # Sort the data by strength
    df.sort_values('normalized_comp_strength', ascending=False, inplace=True)

    # Split the data into three 33% groups
    one_third_length = int(0.33 * len(df))

    df_strongest_33 = df.iloc[:one_third_length]  # Strongest 33%
    df_middle_33 = df.iloc[one_third_length:2*one_third_length]  # Middle 33%
    df_weakest_33 = df.iloc[2*one_third_length:]  # Weakest 33%

    # Plot all three groups on the same plot with different colors
    plt.scatter(df_strongest_33[0], df_strongest_33[1], color='r', label='Strongest 33%', marker='o')
    plt.scatter(df_middle_33[0], df_middle_33[1], color='mediumpurple', label='Middle 33%', marker='o')
    plt.scatter(df_weakest_33[0], df_weakest_33[1], color='b', label='Weakest 33%', marker='o')

    # Set axis labels and limits
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.xlim(-350, 220)
    plt.ylim(-310, 200)

    # Add a title and a legend to distinguish between the groups
    plt.title('Full Data Set: Strongest, Middle, and Weakest 33%')
    plt.legend()

    # Save the combined plot to a file
    plt.savefig(f'plots/combined_full_data.png')
    plt.show()


def eigenvalue_plot(cumulative_variance, percentile):
    print("___________________________________________________________________________________________________________")
    print("Percentages of explanation for the eigenvectors")
    for i in range(8):
        print(f"Cumulative: {cumulative_variance[i]:.4f} \t Percentage: {percentile[i]:.4f}")
    print("___________________________________________________________________________________________________________")
    # Plot the variance explained
    plt.axhline(y=0.7594, color='black', linestyle='-', label='y=0.7594')
    plt.plot(percentile, 'x-', label="Explained Variance")
    plt.plot(cumulative_variance, 'o--', label="Cumulative Variance")
    plt.xticks(np.arange(len(percentile)), np.arange(1, len(percentile) + 1))
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    plt.legend()
    plt.title('Explained Variance by Principal Components')
    plt.savefig('plots/Explained_variance.png')
    plt.grid()
    plt.show()


def find_coeff(normalized_df, attribute_labels):
    U, S, Vt = np.linalg.svd(normalized_df, full_matrices=False)
    V = Vt.T
    N, M = normalized_df.shape

    pcs = [0, 1, 2]
    legendStrs = ["PC" + str(e + 1) for e in pcs]
    bw = 0.2
    r = np.arange(1, M + 1)
    for i in pcs:
        plt.bar(r + i * bw, V[:, i], width=bw)
    plt.xticks(r + bw, attribute_labels)
    plt.xlabel("Attributes")
    plt.ylabel("Component coefficients")
    plt.legend(legendStrs)
    plt.grid()
    plt.title("PCA Component Coefficients")
    plt.show()

def histograms(df_vectors, attribute_labels):
    num_cols = df_vectors.shape[1]

    _, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()

    for col in range(num_cols):
        col_data = df_vectors[:, col]
        ax = axes[col]

        # Create a histogram without plotting (to get the bins and counts)
        counts, bins, patches = ax.hist(col_data, bins=8, edgecolor='black')

        # Calculate the mean of the column data
        mean = np.mean(col_data)

        # Calculate the bin centers and their distance from the mean
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        distance_from_mean = np.abs(bin_centers - mean)

        # Normalize the distances to map them to a colormap
        norm = plt.Normalize(vmin=distance_from_mean.min(), vmax=distance_from_mean.max())
        colormap = cm.get_cmap('coolwarm')

        # Apply the color based on the distance from the mean
        for dist, patch in zip(distance_from_mean, patches):
            color = colormap(norm(dist))
            patch.set_facecolor(color)

        # Increase the number of x-axis intervals automatically with MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))  # Set maximum number of ticks to 10
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))  # Set maximum number of ticks to 10

        # Rotate the x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        ax.set_xlabel('Value')
        ax.set_ylabel('# of Observations')
        ax.set_title(f'{attribute_labels[col]}')

    plt.tight_layout(pad=2.0)
    plt.show()

def correlation_matrix(normalized_df, attribute_labels):
    df = pd.DataFrame(normalized_df, columns=attribute_labels)
    
    corrMatrix = df.corr()
    
    plt.figure(figsize=(8, 8))
    sns.heatmap(corrMatrix, annot=True, cmap='coolwarm', linewidths=0.5, xticklabels=attribute_labels, yticklabels=attribute_labels)
    plt.title('Correlation Matrix Heatmap')
    plt.subplots_adjust(bottom=0.2)
    plt.show()
