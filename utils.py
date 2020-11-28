import numpy as np
from scipy.spatial import distance
from scipy.spatial.distance import pdist


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on the earth (specified in decimal degrees). 
    Harvesine sklearn distance was not working as expected, that is why this function was added maily extracted from:
    https://stackoverflow.com/questions/29545704/fast-haversine-approximation-python-pandas 
    Input is two lat-lon observations.

    Args:
        lat1 : Latitude 1 of the observation.
        lon1 : Longitude 1 of the observation.
        lat2 : Latitude 2 of the observation.
        lon2 : Longitude 2 of the observation.
    
    Returns:
        Harvesine distance in KM.
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    
    c = 2 * np.arcsin(np.sqrt(a))
    km = (6371000/1000) * c
    return km

def euclidean_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the euclidean distance between two pair of coordinates.
    Args:
        lat1 : Latitude 1 of the observation.
        lon1 : Longitude 1 of the observation.
        lat2 : Latitude 2 of the observation.
        lon2 : Longitude 2 of the observation.
    
    Returns:
        Euclidean distance
    """
    euclidean = distance.euclidean([lat1, lon1], [lat2, lon2])
    return(euclidean)

def manhattan_distance(lat1, lon1, lat2, lon2):
    """
    Calculates the manhattan distance between two coordinates (remember is the sum of the absolute difference of the coordinates).
    Args:
        lat1 : Latitude 1 of the observation.
        lon1 : Longitude 1 of the observation.
        lat2 : Latitude 2 of the observation.
        lon2 : Longitude 2 of the observation.
    Returns:
        Manhattan distance.
    """
    manhattan_dist = np.abs(lat1 - lat2) + np.abs(lon1 - lon2)
    return(manhattan_dist)
        
    
from math import radians
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

df["lat_radians"] = df["lat"].apply(lambda x: radians(x))
df["lon_radians"] = df["lon"].apply(lambda x: radians(x))

#haversine distance is on km scale
dist_matrix_harvesine = (pairwise_distances(df[["lat_radians", "lon_radians"]], metric='haversine'))*(6371000/1000)
dist_matrix_euclidean = pairwise_distances(df[["lat", "lon"]], metric='euclidean')
dist_matrix_euclidean = pairwise_distances(df[["lat", "lon"]], metric='manhattan')   