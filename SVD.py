import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_orig = pd.read_excel('concrete+compressive+strength/Concrete_Data.xls')
df = df_orig.loc[:, df_orig.columns != 'Concrete compressive strength(MPa, megapascals) ']

#normalizing the vectors / subtracting the mean
row_count = len(df.columns)
df_vectors = df.to_numpy()
average_vector = np.mean(df_vectors, axis=0)
normalized_df = df_vectors - average_vector

#Performing SVD using NumPy
U, S, eigenvectors = np.linalg.svd(normalized_df, full_matrices=False)

#to visuallize the usefulness of each eigenvector
eigen_sum = np.sum(S ** 2)
percentile = (S ** 2) / eigen_sum

print("___________________________________________________________________________________________________________")
print("Percenteges of the explaination for the eigenvectors")
for i in range(0, 8):
    sum = 0
    for j in range(0, i):
        sum += percentile[j]
    print(str(sum) + "\t\t" + str(percentile[i]))
print("___________________________________________________________________________________________________________")
plt.plot(percentile)
plt.show()

print(S.shape)
print(U.shape)
print(eigenvectors.shape)

#projection onto 3d
basis_vectors = eigenvectors[0:3]
projected_data = np.dot(normalized_df, np.transpose(basis_vectors))
x = projected_data[:, 0]
y = projected_data[:, 1]
z = projected_data[:, 2]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Projection of Data Using Covariance Method')
plt.show()

#projected onto 2d space
basis_vectors = eigenvectors[0:2]
projected_data = np.dot(normalized_df, np.transpose(basis_vectors))
x = projected_data[:, 0]
y = projected_data[:, 1]
plt.scatter(x, y)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('2D Projection of Data Using Covariance Method')
plt.show()

