"""
In this assignment you should interpolate the given function.
"""

import sampleFunctions
import numpy as np

# leaderboard 12/2:
def get_cubic(a, b, c, d):
    return lambda t: np.power(1 - t, 3) * a + 3 * np.power(1 - t, 2) * t * b + 3 * (1 - t) * np.power(t, 2) * c + np.power(t, 3) * d


class Assignment1:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        starting to interpolate arbitrary functions.
        """

        pass

    def interpolate(self, f: callable, a: float, b: float, n: int) -> callable:
        """
        Interpolate the function f in the closed range [a,b] using at most n 
        points. Your main objective is minimizing the interpolation error.
        Your secondary objective is minimizing the running time. 
        The assignment will be tested on variety of different functions with 
        large n values. 
        
        Interpolation error will be measured as the average absolute error at 
        2*n random points between a and b. See test_with_poly() below. 

        Note: It is forbidden to call f more than n times. 

        Note: This assignment can be solved trivially with running time O(n^2)
        or it can be solved with running time of O(n) with some preprocessing.
        **Accurate O(n) solutions will receive higher grades.** 
        
        Note: sometimes you can get very accurate solutions with only few points, 
        significantly less than n. 
        
        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        n : int
            maximal number of points to use.

        Returns
        -------
        The interpolating function.
        """
        c_1 = np.ones(n - 2)
        b_1 = 4 * np.ones(n - 1)
        a_1 = np.ones(n - 2)
        b_1[0] = 2
        b_1[-1] = 7
        a_1[-1] = 2
        x_values = np.linspace(a, b, n)
        y_values = []
        for x_i in x_values:
            y_values.append(f(x_i))

        stack_mat = np.stack((x_values, y_values), axis=1)
        stack_calc_mat = np.zeros((len(stack_mat) - 1, 2))
        for i in range(len(stack_mat) - 1):
            stack_calc_mat[i] = 4 * stack_mat[i] + 2 * stack_mat[i + 1]

        stack_calc_mat[0] = stack_mat[0] + 2 * stack_mat[1]

        stack_calc_mat[-1] = 8 * stack_mat[n - 2] + stack_mat[n - 1]

        c_tag = list()
        d_tag = list()
        c_tag.append(c_1[0] / b_1[0])
        d_tag.append(stack_calc_mat[0] / b_1[0])
        for i in range(1, n - 2):
            c_tag.append(c_1[i] / (b_1[i] - a_1[i - 1] * c_tag[i - 1]))
        for i in range(1, n - 1):
            d_tag.append((stack_calc_mat[i] - (a_1[i - 1] * d_tag[i - 1])) / (b_1[i] - a_1[i - 1] * c_tag[i - 1]))
        A = np.zeros_like(d_tag)
        A[-1] = d_tag[-1]
        for i in range(len(d_tag) - 2, -1, -1):
            A[i] = d_tag[i] - (c_tag[i] * A[i + 1])
        B = np.zeros_like(A)
        for i in range(len(stack_mat) - 2):
            B[i] = 2 * stack_mat[i + 1] - A[i + 1]
        B[len(stack_mat) - 2] = (A[len(stack_mat) - 2] + stack_mat[len(stack_mat) - 2]) / 2
        B[0] = 2 * A[0] - stack_mat[0]

        my_splines = {}
        for i in range(len(stack_mat) - 1):
            my_splines[(stack_mat[i][0], stack_mat[i + 1][0])] = get_cubic(stack_mat[i], A[i], B[i], stack_mat[i + 1])

        def interpolation_to_return(x):
            for i in my_splines:
                if x >= i[0] and x <= i[1]:
                    t = (x - i[0]) / (i[1] - i[0])
                    return my_splines[i](t)[1]
        return interpolation_to_return




##########################################################################


import unittest
from functionUtils import *
from tqdm import tqdm


class TestAssignment1(unittest.TestCase):

    def test_with_poly(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0

        d = 30
        for i in tqdm(range(100)):
            a = np.random.randn(d)
            a = np.random.randn(d)

            f = np.poly1d(a)

            ff = ass1.interpolate(f, 0, 5, 10)

            xs = np.random.random(200)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / 200
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print(T)
        print(mean_err)

    def test_with_poly_restrict(self):
        ass1 = Assignment1()
        a = np.random.randn(5)
        f = RESTRICT_INVOCATIONS(10)(np.poly1d(a))
        ff = ass1.interpolate(f, 0, 5, 10)
        xs = np.random.random(20)
        for x in xs:
            yy = ff(x)


if __name__ == "__main__":
    unittest.main()
