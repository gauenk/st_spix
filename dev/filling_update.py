
"""

   Faster filling

"""

import numpy as np
import torch as th
import itertools
from itertools import permutations


def generate_binary_grid(n=3):
    # 2^(n*n) combos of 0 & 1
    lst = list(itertools.product([0, 1], repeat=n*n))
    lst = 2*np.array(lst).reshape(-1,n,n)-1
    return lst

def main():
    lst = generate_binary_grid(3)
    print(lst.shape)
    ix = 217+135
    grid = lst[ix]
    print(grid)


if __name__ == "__main__":
    main()
