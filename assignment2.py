"""
In this assignment you should find the intersection points for two functions.
"""

import numpy as np
import time
import random
from collections.abc import Iterable


class Assignment2:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def intersections(self, f1: callable, f2: callable, a: float, b: float, maxerr=0.001) -> Iterable:
        """
        Find as many intersection points as you can. The assignment will be
        tested on functions that have at least two intersection points, one
        with a positive x and one with a negative x.
        
        This function may not work correctly if there is infinite number of
        intersection points. 


        Parameters
        ----------
        f1 : callable
            the first given function
        f2 : callable
            the second given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        maxerr : float
            An upper bound on the difference between the
            function values at the approximate intersection points.


        Returns
        -------
        X : iterable of approximate intersection Xs such that for each x in X:
            |f1(x)-f2(x)|<=maxerr.

        """

        f = lambda X: f1(X) - f2(X)

        def derivative_calc(f: callable, z):
            h = 0.00000000001
            return (f(z + h) - f(z)) / (2*h)

        def newton_raphson_bisection(f, a, b):
            try:
                a_value = f(a)
                b_value = f(b)
            except:
                # leaderboard 13/2 11:00
                return None

            if a_value * b_value >= 0:
                return None

            curr = (a + b) / 2
            curr_value = f(curr)
            if a_value * curr_value < 0:
                b = curr
            else:
                a = curr
            a_value = f(a)
            b_value = f(b)
            while abs(curr_value) > maxerr:
                # leaderboard 14/2:
                curr_derivative = derivative_calc(f, curr)
                new_nr_curr = a
                if curr_derivative and curr_derivative != 0:
                    new_nr_curr = curr - (curr_value / curr_derivative)
                if a < new_nr_curr < b:
                    curr = new_nr_curr
                    if a_value * f(curr) > 0:
                        a = curr
                    elif b_value * f(curr) > 0:
                        b = curr
                    else:
                        return a if a_value == 0 else b
                    curr_value = f(curr)
                    a_value, b_value = f(a), f(b)
                else:
                    if abs(a - b) < 1e-5:
                        return None
                    curr = (a + b) / 2
                    curr_value = f(curr)
                    if a_value * curr_value < 0:
                        b = curr
                    else:
                        a = curr
                    a_value, b_value = f(a), f(b)
                    curr = (a + b) / 2
                    curr_value = f(curr)
            return curr

        result = []
        x_result = np.linspace(a, b, num=100)
        y_result = [[], []]
        for x_ in x_result:
            try:
                y_result[1].append(f(x_))
                y_result[0].append(x_)
            except ValueError:
                continue
        # leaderboard 13/2 11:00
        # can also give try and except a try
        if y_result:
            for i in range(len(y_result[0]) - 1):
                if y_result[1][i] * y_result[1][i + 1] <= 0:
                    if y_result[1][i] * y_result[1][i + 1] == 0:
                        continue
                    intersection_ = newton_raphson_bisection(f, y_result[0][i], y_result[0][i + 1])
                    # leaderboard 13/2 11:00
                    # if inter:
                    if intersection_ is not None:
                        result.append(intersection_)
        if abs(f(a)) < maxerr:
            result.append(a)
        if abs(f(b)) < maxerr:
            result.append(b)
        return result



##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment2(unittest.TestCase):

    def test_sqr(self):

        ass2 = Assignment2()

        f1 = np.poly1d([-1, 0, 1])
        f2 = np.poly1d([1, 0, -1])

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    def test_poly(self):

        ass2 = Assignment2()

        f1, f2 = randomIntersectingPolynomials(10)

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))


if __name__ == "__main__":
    unittest.main()
