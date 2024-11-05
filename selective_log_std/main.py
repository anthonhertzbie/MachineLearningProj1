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

# Log normalize specific columns by index (2nd, 3rd, and 8th columns are at indices 1, 2, and 7)
df.iloc[:, [1, 2, 7]] = df.iloc[:, [1, 2, 7]].apply(lambda x: np.log(x + 1))

# Min-max normalize the remaining columns
# Select columns excluding 1, 2, and 7 for min-max normalization
df = df.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
df = (df * 2 - 1)


# Calculating the mean and standard deviation of each attribute
sd_vector = np.std(df, axis=0)

# Standardizing the log-normalized data
df = (df) / sd_vector

df = df.to_numpy()

#----------------------------------------------------------------
# Standardized
#----------------------------------------------------------------
visualization.find_coeff(df, attribute_labels)
visualization.histograms(df_pre, attribute_labels)
visualization.histograms(df, attribute_labels)
visualization.correlation_matrix(df_pre, attribute_labels)
#COV.perform_cov(normalized_df, strength_norm)
SVD.perform_svd(df, strength_norm)