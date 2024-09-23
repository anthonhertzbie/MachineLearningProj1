import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

compres_strength = 'Concrete compressive strength(MPa, megapascals) '

df_orig = pd.read_excel('concrete+compressive+strength/Concrete_Data.xls')
df = df_orig.loc[:, df_orig.columns != compres_strength]

#normailzing strength for vizualisation
strength_norm = ((df_orig[compres_strength] - df_orig[compres_strength].min()) /
            (df_orig[compres_strength].max() - df_orig[compres_strength].min()))

#normalizing the vectors / subtracting the mean
row_count = len(df.columns)
df_vectors = df.to_numpy()
average_vector = np.mean(df_vectors, axis=0)
normalized_df = df_vectors - average_vector

#Performing SVD using NumPy
U, S, Vt = np.linalg.svd(normalized_df, full_matrices=False)
eigenvectors = Vt.T
print(eigenvectors)

#to visuallize the usefulness of each eigenvector
eigen_sum = np.sum(S ** 2)
percentile = (S ** 2) / eigen_sum

# Cumulative variance explained
cumulative_variance = np.cumsum(percentile)

print("___________________________________________________________________________________________________________")
print("Percentages of explanation for the eigenvectors")
for i in range(8):
    print(f"Cumulative: {cumulative_variance[i]:.4f} \t Percentage: {percentile[i]:.4f}")
print("___________________________________________________________________________________________________________")

# Plot the variance explained
plt.plot(percentile, label="Explained Variance")
plt.plot(cumulative_variance, label="Cumulative Variance", linestyle='--')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.legend()
plt.title('Explained Variance by Principal Components')
plt.show()

print(S.shape)
print(U.shape)
print(eigenvectors.shape)

#projection onto 3d
basis_vectors = eigenvectors[:, 0:3]
print(basis_vectors[0])
projected_data = np.dot(normalized_df, basis_vectors)
x = projected_data[:, 0]
y = projected_data[:, 1]
z = projected_data[:, 2]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c=strength_norm, cmap='RdBu')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('3D Projection of Data Using SVD')
plt.show()

#projected onto 2d spa ce
basis_vectors = eigenvectors[:, 0:2]
print(basis_vectors)
projected_data = np.dot(normalized_df, basis_vectors)
x = projected_data[:, 0]
y = projected_data[:, 1]
plt.scatter(x, y, c=strength_norm, cmap='RdBu')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('2D Projection of Data Using SVD')
plt.show()

