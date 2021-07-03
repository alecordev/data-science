"""
Use case: Given an array representing responses or chosen values, check the closest/most similar array from a
defined target array.
"""
import numpy as np
from scipy import spatial


target_solutions = {"one": [2, 3, 1, 2], "two": [2, 1, 3, 1]}

user1 = [1, 2, 2, 2]


def scipy_matching():
    for target, tgt_vector in target_solutions.items():
        result = 1 - spatial.distance.cosine(user1, tgt_vector)
        print(f"{target}: {result:0.4f}")


def numpy_matching():
    for target, tgt_vector in target_solutions.items():
        result = np.dot(user1, tgt_vector) / (
            np.linalg.norm(user1) * np.linalg.norm(tgt_vector)
        )
        print(f"{target}: {result:0.4f}")


if __name__ == "__main__":
    scipy_matching()
    numpy_matching()
