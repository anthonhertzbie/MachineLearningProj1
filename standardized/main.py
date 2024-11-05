import pandas as pd
import numpy as np
import SVD
import visualization

compres_strength = 'Concrete compressive strength(MPa, megapascals) '
df_orig = pd.read_excel('../concrete+compressive+strength/Concrete_Data.xls')
df = df_orig.loc[:, df_orig.columns != 'Concrete compressive strength(MPa, megapascals) ']

attribute_labels = ["Cement", "B.F. Slag", "Fly Ash", "Water", "Superplast.", "Coarse Aggr.", "Fine Aggr", "Age"]


#normailzing strength for vizualisation
strength_norm = ((df_orig[compres_strength] - df_orig[compres_strength].min()) /
                 (df_orig[compres_strength].max() - df_orig[compres_strength].min()))

#normalizing the vectors / subtracting the mean
df_pre = df.to_numpy()
df = df.to_numpy()
average_vector = np.mean(df, axis=0)
sd_vector = np.std(df, axis=0)
standardized_df = (df - average_vector) / sd_vector


#----------------------------------------------------------------
# Standardized
#----------------------------------------------------------------
visualization.find_coeff(standardized_df, attribute_labels)
visualization.histograms(df_pre, attribute_labels)
visualization.correlation_matrix(df_pre, attribute_labels)
SVD.perform_svd(standardized_df, strength_norm)