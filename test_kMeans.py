from kMeans import *


def main():
    """

    Returns:

    """
    data = np.genfromtxt('iris.csv', delimiter=',')

    # petal length and petal width
    xyData = data[1:, 2:4]
    # print(xyData)

    clustering = KMeansCluster(xyData, 3, 10)
    clustering.k_means()


if __name__ == '__main__':
    main()