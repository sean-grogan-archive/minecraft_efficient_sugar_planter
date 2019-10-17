from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from docplex.mp.model import Model


def build_constraints(n, m, mdl, X):
    for i in range(n):
        for j in range(m):
            rhs = []
            for p in [i - 1, i + 1]:
                try:
                    rhs.append(mdl.linear_expr(1 - X[p, j]))
                except KeyError:
                    pass
            for q in [j - 1, j + 1]:
                try:
                    rhs.append(mdl.linear_expr(1 - X[i, q]))
                except KeyError:
                    pass
            mdl.add_constraint(X[i, j] <= mdl.sum(ele for ele in rhs))

def efficient_sugar(n, m):
    mdl = Model("Sugar")
    X = mdl.binary_var_matrix(keys1=range(n), keys2=range(m), name="x")
    mdl.maximize(mdl.sum(X[i] for i in X))
    _constraints(n, m, mdl, X)
    mdl.export_as_lp("sugar.lp")
    mdl.solve(log_output=False)
    k = 0
    print(f'Number of sugar canes {k}')
    output_grid(n, m, X)
    return k


def output_grid(num_n, num_m, X):
    myarray = np.zeros((num_n, num_m))
    fig, ax = plt.subplots()
    for i in range(num_n):
        for j in range(num_m):
            if X[i, j].solution_value > 0.9:
                myarray[i, j] = X[i, j].solution_value
    pcm = ax.pcolormesh(myarray, cmap='RdBu_r')
    fig.colorbar(pcm, ax=ax)
    plt.title("Sugar Plantation Map")
    plt.xlabel("x_blocks")
    plt.ylabel("y_blocks")
    ax.set_aspect("equal")
    plt.savefig(f'n({num_n}) by m({num_m}).png')
    plt.close()


if __name__ == '__main__':
    info = defaultdict(dict)
    mx = 20
    for _i in range(2, mx):
        for _j in range(_i, mx):
            print(f'i={_i}, j={_j}')
            info[_i][_j] = efficient_sugar(_i, _j)
            info[_j][_i] = info[_i][_j]
    myarray = np.zeros((mx, mx))
    for key1, row in info.items():
        for key2, value in row.items():
            myarray[key1, key2] = value/(key1*key2)
    fig, ax = plt.subplots()
    pcm = ax.pcolormesh(myarray, cmap='RdBu_r')
    fig.colorbar(pcm, ax=ax)
    plt.title("Sugar Efficiency Curve")
    plt.xlabel("x_blocks")
    plt.ylabel("y_blks")
    plt.show()
    plt.close()
    fig, ax = plt.subplots()
    plt.title("Sugar Efficiency Curve")
    plt.xlabel("n_blocks")
    plt.ylabel("Pct Sugar")
    x, y = list(), list()
    mx_item = None
    mx_val = 0
    for key1, row in info.items():
        for key2, value in row.items():
            x.append(key1*key2)
            y.append(value/(key1*key2))
            if value/(key1*key2) > mx_val:
                mx_item = key1*key2
                mx_val = value/(key1*key2)
    ax.scatter(x, y)
    ax.scatter(mx_item, mx_val)
    plt.show()
    plt.close()


