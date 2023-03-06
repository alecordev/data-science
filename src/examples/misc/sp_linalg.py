"""
Linear System Solving
"""

import numpy as np
from scipy.linalg import solve


def simple():
    """
    Example of solving the following system of equations:

        3 x1 + 2 x2 = 12
        2 x1 - 3 x2 = 1

        x1 = 2
        x2 = 3
    """
    A = np.array(
        [
            [3, 2],
            [2, -1],
        ]
    )

    b = np.array([12, 1]).reshape((2, 1))
    x = solve(A, b)
    print(x)


def larger():
    """
    Consider that a balanced diet should include the following:

    170 units of vitamin A
    180 units of vitamin B
    140 units of vitamin C
    180 units of vitamin D
    350 units of vitamin E

    Food	Vitamin A	Vitamin B	Vitamin C	Vitamin D	Vitamin E
    #1	1	10	1	2	2
    #2	9	1	0	1	1
    #3	2	2	5	1	2
    #4	1	1	1	2	13
    #5	1	1	1	9	2
    """
    A = np.array(
        [
            [1, 9, 2, 1, 1],
            [10, 1, 2, 1, 1],
            [1, 0, 5, 1, 1],
            [2, 1, 1, 2, 9],
            [2, 1, 2, 13, 2],
        ]
    )

    b = np.array([170, 180, 140, 180, 350]).reshape((5, 1))

    x = solve(A, b)
    print(x)


if __name__ == '__main__':
    simple()
    larger()
