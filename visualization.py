import numpy as np
import matplotlib.pyplot as plt


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
    ax.set_title(title)
    plt.savefig(f'plots/{title}.png')
    plt.show()
    return basis_vectors


def project_to2d(title, cbar_label, normalized_df, eigenvectors, strength_norm):
    basis_vectors = eigenvectors[:, 0:2]
    projected_data = np.dot(normalized_df, basis_vectors)
    x = projected_data[:, 0]
    y = projected_data[:, 1]
    plt.scatter(x, y, c=strength_norm, cmap='RdBu_r')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(title)
    cbar = plt.colorbar()
    cbar.set_label(cbar_label)
    plt.savefig(f'plots/{title}.png')
    plt.show()


def eigenvalue_plot(cumulative_variance, percentile):
    print("___________________________________________________________________________________________________________")
    print("Percentages of explanation for the eigenvectors")
    for i in range(8):
        print(f"Cumulative: {cumulative_variance[i]:.4f} \t Percentage: {percentile[i]:.4f}")
    print("___________________________________________________________________________________________________________")
    # Plot the variance explained
    plt.plot(percentile, 'x-', label="Explained Variance")
    plt.plot(cumulative_variance, 'o--', label="Cumulative Variance")
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

    pcs = [0, 1]
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

        ax.hist(col_data, bins=8, edgecolor='black') 
        ax.set_xlabel('Value')
        ax.set_ylabel('Number of Observations')
        ax.set_title(f'Histogram of Attribute {attribute_labels[col]}')
        
    plt.tight_layout(pad=2.0)
    plt.show()