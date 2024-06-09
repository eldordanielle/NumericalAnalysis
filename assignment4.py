"""
In this assignment you should fit a model function of your choice to data 
that you sample from a given function. 

The sampled data is very noisy so you should minimize the mean least squares 
between the model you fit and the data points you sample.  

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You 
must make sure that the fitting function returns at most 5 seconds after the 
allowed running time elapses. If you take an iterative approach and know that 
your iterations may take more than 1-2 seconds break out of any optimization 
loops you have ahead of time.

Note: You are NOT allowed to use any numeric optimization libraries and tools 
for solving this assignment. 

"""

import numpy as np
import time
import random
import torch
import matplotlib.pyplot as plt
import operator as op
from numpy import polynomial
import scipy as sc
from functools import reduce


class Assignment4:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def fit(self, f: callable, a: float, b: float, d: int, maxtime: float) -> callable:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape. 
        
        Parameters
        ----------
        f : callable. 
            A function which returns an approximate (noisy) Y value given X. 
        a: float
            Start of the fitting range
        b: float
            End of the fitting range
        d: int 
            The expected degree of a polynomial matching f
        maxtime : float
            This function returns after at most maxtime seconds. 

        Returns
        -------
        a function:float->float that fits f between a and b
        """

        def solve_equations(AM, BM):
            # leaderboard 12/2:
            if isinstance(AM, int) or isinstance(BM, int):
                return None
            if len(AM) == 0 or len(BM) == 0:
                return None

            for fd in range(len(AM)):
                fdScaler = 0
                if AM[fd][fd] != 0:
                    fdScaler = 1.0 / AM[fd][fd]

                for j in range(len(AM)):
                    AM[fd][j] *= fdScaler

                # f_ = lambda x: x*fdScaler
                # AM[fd] = list(map(f, AM[fd]))

                BM[fd][0] *= fdScaler
                for i in list(range(len(AM)))[0:fd] + list(range(len(AM)))[fd + 1:]:
                    crScaler = AM[i][fd]
                    for j in range(len(AM)):
                        AM[i][j] = AM[i][j] - crScaler * AM[fd][j]
                    BM[i][0] = BM[i][0] - crScaler * BM[fd][0]
            return BM

        def final_polynomial(x):
            deg = deg_least_noisy
            poly = 0
            for sol in x_least_noisy:
                poly += sol[0] * (x ** deg)
                deg -= 1
            return poly

        n = 100 * d
        # check out if the leaderboard returns a better evaluation with
        # 20 (leaderboard = 5.868) or 50 (leaderboard from 12/2 = 8.14)
        # or 100 (leaderboard from 13/2 = 8.66) or 200 (leaderboard from 14/2 = 8.23)
        t1 = time.time()
        points_arr = [(a, f(a))]
        t2 = time.time()
        curr_x = a
        # leaderboard 13/2 11:00
        intervals = 0
        single_interval = float(t2 - t1)
        if single_interval > 0:
            intervals = int(maxtime / single_interval)
        # leaderboard 13/2 11:00
        if n > intervals > 2:  # has to be larger than 2 because we want to call f(a) and f(b) outside the loop
            for i in range(intervals - 2):
                curr_x += (b - a) / (intervals - 2)
                points_arr.append((curr_x, f(curr_x)))
        else:
            # leaderboard 13/2 11:00
            if n > 2: # has to be larger than 2 because we want to call f(a) and f(b) outside the loop
                for i in range(n - 2):
                    curr_x += (b - a) / (n - 2)
                    points_arr.append((curr_x, f(curr_x)))

        points_arr.append((b, f(b)))
        error = np.inf
        x_least_noisy = []
        T = time.time()
        for deg in range(d, 0, -1):
            A = []
            b = []
            for point in points_arr:
                b.append([point[1]])
                a = []
                for i in range(deg, -1, -1):
                    a.append(point[0] ** i)
                A.append(a)

            # leaderboard 12/2
            A = np.asarray(A)
            b = np.asarray(b)

            AT = np.transpose(A)
            ATA = np.matmul(AT, A)
            ATb = np.matmul(AT, b)
            # try:
            #     ATA_inv = np.linalg.inv(np.mat(ATA))
            #     ATA_invATA = np.matmul(ATA_inv, ATA)
            #     ATA_invATb = np.matmul(ATA_inv, ATb)
            #     x_hat = ATA_invATb
            # except:
            #     ATA_T_ATA = np.matmul(np.transpose(ATA), ATA)
            #     ATA_T_ATb = np.matmul(np.transpose(ATA), ATb)
            #     ATA_T_ATA_inv = np.linalg.inv(np.mat(ATA_T_ATA))
            #     x_hat = np.matmul(ATA_T_ATA_inv, ATA_T_ATb)

            x_sol = solve_equations(ATA, ATb)

            if x_sol is not None:
            # try:
                Ax_sol = np.matmul(A, x_sol)
                mis = sum((b[i][0] - Ax_sol[i][0]) ** 2 for i in range(len(b)))
                if mis < error:
                    error = mis
                    x_least_noisy = x_sol
            # except:
            #     break
            if time.time() - T > 0.98 * maxtime:
                # check out what comes out of the next leaderboard
                # 0.88 = 1.944, 0.93 = 2.711, 0.95 = 5.868, 0.98 = 8.14, 0.98 = 8.66
                break

        deg_least_noisy = 0

        try:
            deg_least_noisy = len(x_least_noisy) - 1
        except:
            deg_least_noisy = 0

        return final_polynomial


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment4(unittest.TestCase):

    def test_return(self):
        f = NOISY(0.01)(poly(1, 1, 1))
        ass4 = Assignment4()
        T = time.time()
        shape = ass4.fit(f=f, a=1, b=5, d=50, maxtime=20)
        T = time.time() - T
        self.assertLessEqual(T, 5)

    def test_delay(self):
        f = DELAYED(1)(NOISY(0.01)(poly(1, 1, 1)))

        ass4 = Assignment4()
        T = time.time()
        shape = ass4.fit(f=f, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        self.assertGreaterEqual(T, 5)

    def test_err(self):
        f = poly(1, 1, 1)
        nf = NOISY(1)(f)
        ass4 = Assignment4()
        T = time.time()
        ff = ass4.fit(f=nf, a=0, b=1, d=10, maxtime=5)
        T = time.time() - T
        mse = 0
        for x in np.linspace(0, 1, 1000):
            self.assertNotEquals(f(x), nf(x))
            mse += (f(x) - ff(x)) ** 2
        mse = mse / 1000
        print(mse)


if __name__ == "__main__":
    unittest.main()
