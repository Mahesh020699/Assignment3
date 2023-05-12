import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def cluster_points():
    
    
    """
    Perform clustering on population growth data for selected countries.

    Reads the climate_change.csv file, filters the data for Ireland and India and population growth indicator.
    Performs K-means clustering on the selected data and visualizes the results.

    Returns:
    None
    """
    
    
    dset = pd.read_csv("climate_change.csv", skiprows=3, usecols=lambda col: col != 'Unnamed: 66')
    dset.columns = [str(col).strip().replace('\n', '') for col in dset.columns]

    con = ['Ireland', 'India']
    Ind = ["Population growth (annual %)"]
    dset = dset[dset["Country Name"].isin(con)]
    dset = dset[dset["Indicator Name"].isin(Ind)]

    dset = dset[['Country Name', '1990', '1995', '2000', '2005', '2010', '2015', '2020']]

    dset = dset.set_index('Country Name')
    result = dset.to_numpy()
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(result)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    color = ['#ADD8E6', '#90EE90']

    for iter1, iter2 in enumerate(con):
        plt.scatter(result[iter1, :], result[iter1, :], c=color[iter1], label=iter2)

    plt.scatter(result[:, 0], result[:, 1], c=labels)
    plt.scatter(centers[:, 0], centers[:, 1], marker='D', s=100, c='r')
    plt.scatter(result[:, 0], result[:, 1], c=labels)
    plt.xlabel('From Year 1990')
    plt.ylabel('To Year 2020')
    plt.title('Cluster population Growth comparison by Country', fontweight="bold")
    plt.savefig("cluster_con.png")
    plt.legend()
    plt.show()


def cluster_countries():
    
    
    """
    Perform clustering on population growth data for selected countries.

    Reads the climate_change.csv file, filters the data for selected countries and population growth indicators.
    Performs K-means clustering on the selected data and visualizes the results for each cluster.

    Returns:
    None
    """
    
    
    dset = pd.read_csv("climate_change.csv", skiprows=3, usecols=lambda col: col != 'Unnamed: 66')
    dset.columns = [str(col).strip().replace('\n', '') for col in dset.columns]
    cons = ["United Kingdom", "India", "United States", "Germany"]
    indicators = ["Population growth (annual %)", 'Population, total']
    dset = dset[dset["Country Name"].isin(cons)]
    dset = dset[dset["Indicator Name"].isin(indicators)]

    dset = dset[['Country Name', '1980', '1985', '1990', '1995', '2000', '2005', '2010', '2015', '2020']]
    dset = dset.set_index('Country Name')
    data = dset.to_numpy()
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(data)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    color = ['b', 'g', 'r', 'c']

    for cluster_label in range(len(np.unique(labels))):
        cluster_data = data[labels == cluster_label]
        cons = dset.index[labels == cluster_label]

        plt.figure()
        for i in range(len(cons)):
            plt.plot(cluster_data[i, :], color=color[i], label=cons[i])

        plt.plot(centers[cluster_label, :], color='k', linestyle='--', label='Cluster Center')
        plt.xlabel('Year')
        plt.ylabel('Rate of population increase')
        plt.title('Cluster {} - Rate of population increase by Country'.format(cluster_label), fontweight="bold")
        plt.legend()
        plt.show()


cluster_points()
cluster_countries()