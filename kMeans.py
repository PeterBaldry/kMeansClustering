
import numpy as np
from scipy.spatial import distance
import pandas as pd
import matplotlib.pyplot as plt
import csv


def k_means_cluster(data, numClusters):
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
	
	plot_clusters(clusterCentres, dataX, dataY)

	for i in range(10):
		cluster_centres = update_cluster_centres(clusterCentres, data)
		plot_clusters(clusterCentres, dataX, dataY)


def plot_clusters(clusterCentres, dataX, dataY):
	"""
	"""
	for clusterCentre in clusterCentres:
		plt.scatter(dataX, dataY, c='black')
		plt.scatter(clusterCentre[0],clusterCentre[1], c = 'red')
	plt.show()


def update_cluster_centres(clusterCentres, data):
	"""
	"""

	# list of lists. the inner lists will contain the datapoints
	# that belong to each cluster centre
	clusterData = []

	for dataPoint in data:

		distances = []

		for clusterCentre in clusterCentres:

			dist = distance.euclidean(dataPoint, clusterCentre)
			distances.append(dist)

		closestCentreIndex = np.argmin(distances)
		clusterData.append([closestCentreIndex, dataPoint])

	clusterData = np.array(clusterData)
	new_centres = find_averages(clusterData, clusterCentres)

	return new_centres


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

	data = np.genfromtxt('iris.csv', delimiter = ',')

	#petal length and petal width
	xyData = data[:,2:4]
	

	k_means_cluster(xyData, 3)





if __name__ == '__main__':
	main()
