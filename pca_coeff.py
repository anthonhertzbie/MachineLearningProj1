import matplotlib.pyplot as plt
from scipy.linalg import svd
import numpy as np


def find_coeff(normalized_df, attribute_labels):
    U, S, Vt = np.linalg.svd(normalized_df, full_matrices=False)
    V = Vt.T
    N, M = normalized_df.shape

    pcs = [0, 1, 2]
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
    plt.title("NanoNose: PCA Component Coefficients")
    plt.show()