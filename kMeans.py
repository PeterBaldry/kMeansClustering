"""
Author: Peter Baldry

Two dimensional k-means clustering
"""

import numpy as np
from scipy.spatial import distance
import pandas as pd
import matplotlib.pyplot as plt
import csv
from const import *


class KMeansCluster:

    def __init__(self, data, numClusters, iterations):
        self.data = data
        self.numClusters = numClusters
        self.iterations = iterations

    def k_means(self):
        """
        Performs k means clustering for 2 dimensional data

        Returns: the cluster centre coordinates for all k clusters

        """

        dataX = self.data[:, 0]
        dataY = self.data[:, 1]

        minX, maxX = self.get_min_max(dataX)
        minY, maxY = self.get_min_max(dataY)

        clusterCentres = self.initiate_cluster_centres(minX, maxX, minY, maxY)

        self.plot_clusters(clusterCentres, 0)

        for i in range(self.iterations):
            clusterCentres = self.update_cluster_centres(clusterCentres)
            self.plot_clusters(clusterCentres, i + 1)

        return clusterCentres

    def plot_clusters(self, clusterCentres, iteration):
        """
		Plots the cluster centres and data

        Args:
            clusterCentres: list of [x,y] coordinates for each of the cluster centres
            iteration: the current iteration the algorithm is up to

        Returns: No return, just plot

        """

        for index, clusterCentre in enumerate(clusterCentres):

            for point in self.data:
                closestCentreIndex = self.get_closest_cluster(point, clusterCentres)
                plt.scatter(point[0], point[1], c=COLOURS[closestCentreIndex], s=10)
            plt.scatter(clusterCentre[0], clusterCentre[1], c=COLOURS[index], marker='^', s=100)

        plt.suptitle('Cluster centres after {} iterations'.format(str(iteration)))
        plt.show()

    def update_cluster_centres(self, clusterCentres):
        """
        Updates the cluster centre coordinates

        Args:
            clusterCentres: list of [x,y] cluster centres

        Returns: a list of the new cluster centres

        """

        # list of lists. the inner lists will contain the datapoints
        # that belong to each cluster centre
        clusterData = []

        for dataPoint in self.data:
            closestCentreIndex = self.get_closest_cluster(dataPoint, clusterCentres)
            clusterData.append([closestCentreIndex, dataPoint])

        clusterData = np.array(clusterData)
        new_centres = self.find_means(clusterData, clusterCentres)

        return new_centres

    def get_closest_cluster(self, dataPoint, clusterCentres):
        """
		Gets the closest cluster centre index to a given point

        Args:
            dataPoint: a data point in the form [x,y]
            clusterCentres: the list cluster centre coordinates in the form [x,y]

        Returns: the index of the closest cluster centre

        """
        distances = []

        for clusterCentre in clusterCentres:
            dist = distance.euclidean(dataPoint, clusterCentre)
            distances.append(dist)

        closestCentreIndex = np.argmin(distances)

        return closestCentreIndex

    def find_means(self, clusterData, clusterCentres):
        """
		Finds the means of the data points corresponding to each cluster centre and updates the cluster centres

        Args:
            clusterData: all of the data with each datapoint's corresponding cluster centre also
            clusterCentres: all of the cluster centre coordinates

        Returns: the new cluster centres (means)

        """

        clusterSumsX = [0] * len(clusterCentres)
        clusterSumsY = [0] * len(clusterCentres)
        counts = [0] * len(clusterCentres)

        for dataPoint in clusterData:
            clusterSumsX[dataPoint[0]] += dataPoint[1][0]
            clusterSumsY[dataPoint[0]] += dataPoint[1][1]
            counts[dataPoint[0]] += 1

        for index in range(len(clusterCentres)):
            if (counts[index] == 0):
                xAv = 0
                yAv = 0
            else:
                xAv = clusterSumsX[index] / counts[index]
                yAv = clusterSumsY[index] / counts[index]
            clusterCentres[index] = [xAv, yAv]

        return clusterCentres

    def initiate_cluster_centres(self, minX, maxX, minY, maxY):
        """
		Initiates the coordinates of the cluster centres prior to algorithm starting

        Args:
            numClusters: the number of clusters to be used
            minX: the minimum x value in the data
            maxX: the maximum x value in the data
            minY: the minimum y value in the data
            maxY: the maximum y value in the data

        Returns: the initial coordinates of the cluster centres

        """

        clusterCentres = []

        for clusterNum in range(self.numClusters):
            x = np.random.uniform(minX, maxX)
            y = np.random.uniform(minY, maxY)
            clusterCentres.append([x, y])

        return clusterCentres

    def get_min_max(self, data):
        """
        Gets the minimum and maximum of a given dataset

        Args:
            data: the data array to find the minimum and maximum

        Returns: the minimum and maximum values of the data

        """

        minDat = min(data)
        maxDat = max(data)

        return minDat, maxDat
