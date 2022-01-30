import numpy as np


def f_e_m(weight):
    for k in range(len(weight)):
        for i in range(0, len(weight[k])):  # len=4
            for j in range(0, len(weight[k][i])):  # i=0,len=32
                for m in range(0, len(weight[k][i][j])):
                    am = np.argmax(weight[k][i][j][m].detach().numpy())
                    bm = np.argmin(weight[k][i][j][m].detach().numpy())
                    for n in range(0, len(weight[k][i][j][m])):
                        if n != am and n != bm:
                            weight[k][i][j][m][n].data *= 0
    return weight
