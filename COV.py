import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import visualization as viz

def perform_cov(normalized_df, strength_norm):
    #calculating eigenvalues and eigenvectors with the covariance matrix
    covariance_matrix = np.cov(normalized_df, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    #sorting the eigenvectors
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    #to visuallize the usefulness of each eigenvector
    eigen_sum = np.sum(sorted_eigenvalues)
    percentile = sorted_eigenvalues / eigen_sum
    cumulative_variance = np.cumsum(percentile)

    viz.eigenvalue_plot(cumulative_variance, percentile)

    viz.project_to3d('3D Projection of Data Using Covariance Method',
                     normalized_df, sorted_eigenvectors, strength_norm)

    viz.project_to2d('2D Projection of Data Using Covariance Method',
                     'normalized of compressive strength',
                     normalized_df, sorted_eigenvectors, strength_norm)
