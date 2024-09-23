import numpy as np
import visualization as viz


def perform_svd(normalized_df, strength_norm):
    #Performing SVD using NumPy
    U, S, Vt = np.linalg.svd(normalized_df, full_matrices=False)
    eigenvectors = Vt.T

    #to visuallize the usefulness of each eigenvector
    eigen_sum = np.sum(S ** 2)
    percentile = (S ** 2) / eigen_sum

    # Cumulative variance explained
    cumulative_variance = np.cumsum(percentile)

    viz.eigenvalue_plot(cumulative_variance, percentile)
    print(eigenvectors)
    viz.project_to3d('3D Projection of Data using SVD',
                     normalized_df, eigenvectors, strength_norm)

    viz.project_to2d('2D Projection of Data Using SVD',
                     'normalized of compressive strength',
                     normalized_df, eigenvectors, strength_norm)



