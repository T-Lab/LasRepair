"""
implement confident learning
read in 2 csv and some hyperparameters
one csv is the newest repaired csv, the other is the repaired csv from the previous iteration
output confident matrix, then the weight matrix for the next iteration
"""

import pandas as pd
import numpy as np

def confident_learning(new_csv, old_csv):
    """
    output a confident matrix
    """
    pass


def uncertainty_matrix(first_n_tokens, logits):
    """
    use the first_tokens of one column to calculate the confident matrix
    modify the uncertainty for better performance
    output the uncertainty matrix, same order as the first_n_tokens
    """
    t_dict = {}
    uncertainty = [0] * len(first_n_tokens)
    n = len(first_n_tokens[0])
    for i in range(len(first_n_tokens)):
        observed = first_n_tokens[i][0]
        if observed not in t_dict:
            t_dict[observed] = []
        t_dict[observed].append(logits[i][0])
    for key in t_dict:
        t_dict[key] = np.mean(t_dict[key])

    # you may modify the uncertainty here for better performance
    # for i in range(len(first_n_tokens)):
    #     for j in range(n):
    #         if first_n_tokens[i][j] not in t_dict:
    #             if j == n-1:
    #                 uncertainty[i] = 1  # not sure what to repair at all
    #             else:
    #                 # here I treat the t_value as 0 for token not in t_dict
    #                 uncertainty[i] = 1 - logits[i][j]  # need to repair a new value, reduce the uncertainty
    #         else:
    #             # fall into CL formula
    #             if logits[i][j] > t_dict[first_n_tokens[i][j]]:
    #                 uncertainty[i] = 1 - logits[i][j]
    #             else:
    #                 continue
    # I did revert the weight, as lower weight should be assigned to low quality data
    for i in range(len(first_n_tokens)):
        for j in range(n):
            if first_n_tokens[i][j] not in t_dict:
                if j == n-1:
                    uncertainty[i] = 0  # not sure what to repair at all
                else:
                    # here I treat the t_value as 0 for token not in t_dict
                    uncertainty[i] = logits[i][j]  # need to repair a new value, reduce the uncertainty
            else:
                # fall into CL formula
                if logits[i][j] > t_dict[first_n_tokens[i][j]]:
                    uncertainty[i] = logits[i][j]
                else:
                    continue

    return uncertainty


def weight_matrix(new_csv, old_csv, threshold):
    pass