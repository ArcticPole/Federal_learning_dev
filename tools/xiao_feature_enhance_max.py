import numpy as np


def f_e_m(weight):
    for k in range(len(weight)):
        for i in range(0, len(weight[k])):  # len=4
            for j in range(0, len(weight[k][i])):  # i=0,len=32
                # for k in range(0, len(weight1[i][j])):  # i=0,j=0,len=1
                # print(weight[k][i][j])
                # am=np.argmax(weight[k][i][j].detach().numpy(),axis=0)  # 横坐标
                # bm=np.argmax(weight[k][i][j].detach().numpy(),axis=1)  # 纵坐标
                #
                # ami = np.argmin(weight[k][i][j].detach().numpy(), axis=0)
                # bmi = np.argmin(weight[k][i][j].detach().numpy(), axis=1)
                for m in range(0, len(weight[k][i][j])):
                    am = np.argmax(weight[k][i][j][m].detach().numpy())
                    bm = np.argmin(weight[k][i][j][m].detach().numpy())
                    for n in range(0, len(weight[k][i][j][m])):
                        # a = np.array([m,n])
                        # for l in range(len(bm)):
                        #     b = np.array([bm[l],am[l]])
                        #     c = np.array([bmi[l],ami[l]])
                        #     print(b, c)
                        #     if any(a!=b) and any(a!=c):
                        #         weight[k][i][j][m][n].data*=0
                        if n != am and n != bm:
                            weight[k][i][j][m][n].data *= 0
                # print(weight[k][i][j])
                # print("aaaa")
    return weight
