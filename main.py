import pandas as pd
from sklearn.preprocessing import StandardScaler
import SVD, visualization

def preprocess(df):
    scaler = StandardScaler()

    df_log_norm_std = scaler.fit_transform(df)
    return pd.DataFrame(df_log_norm_std, columns=df.columns)


attribute_labels = ["Cement", "B.F. Slag", "Fly Ash", "Water", "Superplast.", "Coarse Aggr.", "Fine Aggr", "Age"]
compress_strength = "Concrete compressive strength(MPa, megapascals) "


if __name__ == "__main__":
    # load data
    df_orig = pd.read_excel("concrete+compressive+strength/Concrete_Data.xls")

    #normailzing strength for vizualisation
    strength_norm = ((df_orig[compress_strength] - df_orig[compress_strength].min()) /
                    (df_orig[compress_strength].max() - df_orig[compress_strength].min()))

    # use only attributes for std
    df_attributes = df_orig.loc[:, df_orig.columns != compress_strength]

    # We add 1 to avoid negative numbers
    df_preproc = preprocess(df_attributes)

    print(df_preproc.head())
    # visualization
    visualization.find_coeff(df_preproc, attribute_labels)
    print(df_preproc.head())
    visualization.histograms(df_preproc, attribute_labels)
    print(df_preproc.head())
    visualization.correlation_matrix(df_preproc, attribute_labels)

    #
    SVD.perform_svd(df_preproc, strength_norm)