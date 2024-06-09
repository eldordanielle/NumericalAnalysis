"""
In this assignment you should fit a model function of your choice to data 
that you sample from a contour of given shape. Then you should calculate
the area of that shape. 

The sampled data is very noisy so you should minimize the mean least squares 
between the model you fit and the data points you sample.  

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You 
must make sure that the fitting function returns at most 5 seconds after the 
allowed running time elapses. If you know that your iterations may take more 
than 1-2 seconds break out of any optimization loops you have ahead of time.

Note: You are allowed to use any numeric optimization libraries and tools you want
for solving this assignment. 
Note: !!!Despite previous note, using reflection to check for the parameters 
of the sampled function is considered cheating!!! You are only allowed to 
get (x,y) points from the given shape by calling sample(). 
"""
import os
os.environ["OMP_NUM_THREADS"] = '2'
import math

import numpy as np
import time
import random

import assignment4
from functionUtils import AbstractShape
import sampleFunctions as sf
from scipy.spatial import distance
import assignment3
import torch
import sampleFunctions as sf
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# leaderboard 12/2, 13/2 11:00
def trapzoid (x_0, x_0_val, x_1, x_1_val):
    return (x_0 - x_1) * ((x_0_val + x_1_val) / 2)


class MyShape(AbstractShape):
    # change this class with anything you need to implement the shape

    def __init__(self, sample):
        self.sample = sample
        Mat_ = torch.tensor(
            [[-1, +3, -3, +1],
             [+3, -6, +3, 0],
             [-3, +3, 0, 0],
             [+1, 0, 0, 0]], dtype=torch.double

        )
        self.Mat_ = Mat_
        Mat_2 = torch.tensor(
            [[1, -2, 1],
             [-2, 2, 0],
             [1, 0, 0]], dtype=torch.double
        )
        self.Mat_2 = Mat_2

    def sample(self):
        return self.sample

    def contour(self, n):

        points_arr = [self.sample() for i in range(n)]

        copy_points_arr = points_arr.copy()
        result_ = []
        result_.append(points_arr.pop(0))
        euclidian_distance = []
        for i in range(1, len(copy_points_arr)):
            euclidian_distance = [distance.euclidean(result_[-1], points_arr[j]) for j in range(len(points_arr))]
            a = np.argsort(euclidian_distance)
            result_.append(points_arr.pop(a[0]))
        return result_

    def area(self):
        assignment3_imported = assignment3.Assignment3()
        area = 0
        P = []
        clust = self.contour(300)
        c_tags = clust
        n_clusters = 15
        kmeans_ = KMeans(n_clusters=n_clusters, n_init=10)
        kmeans_.fit(clust)
        cluster_centers = kmeans_.cluster_centers_

        result_ = []
        copy_cluster_centers = cluster_centers.copy()
        cluster_centers = list(cluster_centers)

        result_.append(cluster_centers.pop(0))
        euclidian_distance = []

        for i in range(0, len(cluster_centers)):
            euclidian_distance = [distance.euclidean(result_[-1], cluster_centers[j]) for j in range(len(cluster_centers))]
            a=np.argsort(euclidian_distance)
            result_.append(cluster_centers.pop(a[0]))

        cluster_centers = result_
        n = 1000
        t = torch.tensor(np.linspace(0.0, 1.0, n), dtype=torch.double)
        T2 = torch.stack([t ** 2, t ** 1, t ** 0]).T
        clust_centers = []
        cluster_centers = list(cluster_centers)
        cluster_centers.append(cluster_centers[0])
        clust = torch.tensor(cluster_centers, dtype=torch.double)
        plt.scatter(clust.T[0], clust.T[1])
        curr_clust = cluster_centers[0]

        for i in range(0, n_clusters):
            if i!=n_clusters:
                curr_clust = 2 * clust[i] - curr_clust
                i_clust = torch.stack([
                    clust[i],
                    curr_clust,
                    clust[i + 1],
                ])
            elif i==n_clusters:
                curr_clust = 2 * clust[i] - curr_clust
                i_clust = torch.stack([
                    clust[i],
                    curr_clust,
                    clust[0],
                ])

            def f(temp):
                temp = (temp - i_clust[0][0]) / (i_clust[2][0] - i_clust[0][0])

                mat_temp = torch.tensor([temp ** 2, temp, 1], dtype=torch.double)

                return (mat_temp @ (self.Mat_2) @ i_clust)[1]

            area += assignment3_imported.integrate(f, i_clust[0][0], i_clust[2][0], 1000)

        return abs(area)

        pass


class Assignment5:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def area(self, contour: callable, maxerr=0.001) -> np.float32:
        """
        Compute the area of the shape with the given contour. 

        Parameters
        ----------
        contour : callable
            Same as AbstractShape.contour 
        maxerr : TYPE, optional
            The target error of the area computation. The default is 0.001.

        Returns
        -------
        The area of the shape.

        """

        points_arr = contour(int(2 / maxerr))

        # Green's theorem
        area = 0.5 * np.sum(points_arr[:, 0] * np.roll(points_arr[:, 1], -1) -
                            points_arr[:, 1] * np.roll(points_arr[:, 0], -1))

        return np.float32(abs(area))



    
    def fit_shape(self, sample: callable, maxtime: float) -> AbstractShape:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape. 
        
        Parameters
        ----------
        sample : callable. 
            An iterable which returns a data point that is near the shape contour.
        maxtime : float
            This function returns after at most maxtime seconds. 

        Returns
        -------
        An object extending AbstractShape. 
        """

        # leaderboard 13/2 11:00

        return MyShape(sample=sample)




##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment5(unittest.TestCase):

    def test_return(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertLessEqual(T, 5)

    def test_delay(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)

        def sample():
            time.sleep(7)
            return circ()

        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=sample, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertGreaterEqual(T, 5)

    def test_circle_area(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)

    def test_bezier_fit(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)


if __name__ == "__main__":
    unittest.main()
