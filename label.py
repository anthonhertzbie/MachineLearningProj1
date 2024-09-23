import string

def label(num_cols):
    labels = []
    for i in range(num_cols - 1):
        labels.append(chr((i % 26) + 65))
    labels.append("Age")
    return labels