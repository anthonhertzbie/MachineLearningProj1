import pandas as pd
import numpy as np
import COV
import SVD
import label
import visualization

compres_strength = 'Concrete compressive strength(MPa, megapascals) '
df_orig = pd.read_excel('concrete+compressive+strength/Concrete_Data.xls')
df = df_orig.loc[:, df_orig.columns != 'Concrete compressive strength(MPa, megapascals) ']


attribute_labels = label.label(len(df.columns))


#normailzing strength for vizualisation
strength_norm = ((df_orig[compres_strength] - df_orig[compres_strength].min()) /
                 (df_orig[compres_strength].max() - df_orig[compres_strength].min()))

#normalizing the vectors / subtracting the mean
row_count = len(df.columns)
df_vectors = df.to_numpy()
average_vector = np.mean(df_vectors, axis=0)
normalized_df = (df_vectors - average_vector)


visualization.find_coeff(normalized_df, attribute_labels)
visualization.histograms(df_vectors, attribute_labels)
COV.perform_cov(normalized_df, strength_norm)
SVD.perform_svd(normalized_df, strength_norm)