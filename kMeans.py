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


def k_means_cluster(data, numClusters, iterations):
	"""

	

	Paramaters:
	- data: in the form [(x1,y1), (x2, y2), ..., (xn,yn)]
	- numClusters: integer - the number of clusters 

	Returns:
	- clusterCentres - the coordinates of the cluster centres
	- dataClustered - each data point's corresponding cluster
	"""


	dataX = data[:,0]
	dataY = data[:,1]

	minX, maxX = get_min_max(dataX) 
	minY, maxY = get_min_max(dataY)
	
	clusterCentres = initiate_cluster_centres(numClusters, minX, maxX, minY, maxY)
	
	plot_clusters(clusterCentres, data, 0)

	for i in range(iterations):
		cluster_centres = update_cluster_centres(clusterCentres, data)
		plot_clusters(clusterCentres, data, i+1)


def plot_clusters(clusterCentres, data, iteration):
	"""
	"""
	
	for index, clusterCentre in enumerate(clusterCentres):

		for point in data:
			closestCentreIndex = get_closest_cluster(point, clusterCentres)
			plt.scatter(point[0], point[1], c=COLOURS[closestCentreIndex], s = 10)
		plt.scatter(clusterCentre[0],clusterCentre[1], c = COLOURS[index], marker = '^', s= 100)

	plt.suptitle('Cluster centres after {} iterations'.format(str(iteration)))
	plt.show()


def update_cluster_centres(clusterCentres, data):
	"""
	"""

	# list of lists. the inner lists will contain the datapoints
	# that belong to each cluster centre
	clusterData = []

	for dataPoint in data:

		closestCentreIndex = get_closest_cluster(dataPoint, clusterCentres)
		clusterData.append([closestCentreIndex, dataPoint])


	clusterData = np.array(clusterData)
	new_centres = find_averages(clusterData, clusterCentres)

	return new_centres

def get_closest_cluster(dataPoint, clusterCentres):

		distances = []

		for clusterCentre in clusterCentres:

			dist = distance.euclidean(dataPoint, clusterCentre)
			distances.append(dist)

		closestCentreIndex = np.argmin(distances)

		return closestCentreIndex
		

def find_averages(clusterData, clusterCentres):
	"""
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
			xAv = clusterSumsX[index]/counts[index]
			yAv = clusterSumsY[index]/counts[index]
		clusterCentres[index] = [xAv, yAv]

	return clusterCentres

	

def initiate_cluster_centres(numClusters, minX, maxX, minY, maxY):
	"""
	"""

	clusterCentres = []

	for clusterNum in range(numClusters):

		x = np.random.uniform(minX, maxX)
		y = np.random.uniform(minY, maxY)
		clusterCentres.append([x,y])

	return clusterCentres


def get_min_max(data):
	"""
	"""


	minDat = min(data)
	maxDat = max(data)

	return minDat, maxDat



def main():
	"""
	"""
	data = np.genfromtxt('iris.csv', delimiter = ',')

	#petal length and petal width
	xyData = data[1:,2:4]
	print(xyData)
	

	k_means_cluster(xyData, 3, 10)





if __name__ == '__main__':
	main()
