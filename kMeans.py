
import numpy as np
from scipy.spatial import distance




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




def update_cluster_centres(clusterCentres, data):
	"""
	"""

	clusterData = []

	for dataPoint in data:

		distances = []

		for clusterCentre in clusterCentres:

			dist = distance.euclidean(dataPoint, clusterCentre)
			distances.append(dist)

		closestCentreIndex = np.argmin(distances)
		clusterData.append([datapoint, closestCentreIndex])

	



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



def main()