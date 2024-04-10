import pandas as pd

def cal_loss_weight(df):
    label_names = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
    total = len(df)
    pos_weight = []
    for label in label_names:
        positives = int(df.loc[:, [label]].sum().iloc[0])
        negatives = total - positives
        weight = negatives / positives
        pos_weight.append(weight)

    return pos_weight