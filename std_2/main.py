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


df_pre = df.to_numpy()

# Normalize the first 7 based on their individual means
mean = df.iloc[:, :7].mean()

# Standardize the first 7 based on their collective std deviation
std_7 = df.iloc[:, :7].to_numpy().flatten().std()
df.iloc[:, :7] = (df.iloc[:, :7] - mean) / std_7

# Standardize the 8th attribute using its own mean & std-dev
df.iloc[:, 7] = (df.iloc[:, 7] - df.iloc[:, 7].mean()) / df.iloc[:, 7].std()

df = df.to_numpy()

#----------------------------------------------------------------
# Standardized
#----------------------------------------------------------------
visualization.find_coeff(df, attribute_labels)
visualization.histograms(df_pre, attribute_labels)
visualization.histograms(df, attribute_labels)
visualization.correlation_matrix(df, attribute_labels)
SVD.perform_svd(df, strength_norm)