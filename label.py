import string

def label(num_cols):
    labels = []
    for i in range(num_cols - 1):
        label = ""
        while i >= 0:
            label = chr((i % 26) + 65) + label  # Convert to ASCII uppercase letters
            i = i // 26 - 1
        labels.append(label)
    labels.append("Age")
    return labels