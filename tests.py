import pandas as pd
from math import radians
from scipy.spatial.distance import cdist
from scipy.spatial import distance_matrix
import numpy as np
from utils import haversine_distance, manhattan_distance
import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose

def test_balanced_stops(transformed_df, balance_base = "default"):
    if balance_base == "default":
        base = 3977
    else:
        base = transformed_df.shape[0]
    balanced_stops = transformed_df.iloc[:, 1:].sum()
    balance_per = balanced_stops / base
    deviation = ((balance_per - (1/6))**2).sum()/6
    return(deviation)

def test_balanced_distributions(transformed_df, main_df):
    assert transformed_df.shape[0] == main_df.shape[0]
    joined_df = transformed_df.join(main_df[["Id_Cliente", "Vol_Entrega", "Frecuencia"]], on="Id_Cliente", how = "inner", rsuffix="_transformed")
    joined_df.drop("Id_Cliente_transformed" ,axis = 1, inplace = True)
    base = (joined_df.Frecuencia * joined_df.Vol_Entrega).sum()
    zones = ["D1", "D2", "D3", "D4", "D5", "D6"]
    zone_volumes = []
    for zone in zones:
        zone_volumes.append((((joined_df[zone] * joined_df["Vol_Entrega"]).sum() / base) - (1/6))**2)
    deviation = sum(zone_volumes) / 6
    return(deviation)

def distance_intra_cluster(transformed_df, main_df, centroids):
    zones = ["D1", "D2", "D3", "D4", "D5", "D6"]
    centroids_map = dict(zip(zones, centroids))
    centroids_df = pd.DataFrame.from_dict(centroids_map, orient = "index", columns = ["lat_centroids", "lon_centroids"])
    centroids_df = centroids_df.reset_index().rename(columns = {"index":"zones"})
    
    unpivoted_df = pd.melt(transformed_df, id_vars=['Id_Cliente'], var_name= "zones")
    unpivoted_df = unpivoted_df[unpivoted_df.value>0]
    joined_df = unpivoted_df.merge(main_df[["Id_Cliente", "lat", "lon"]], on="Id_Cliente", how = "left")\
                            .merge(centroids_df,  on="zones", how = "left")
    joined_df["manhattan"] = manhattan_distance(joined_df["lat"], joined_df["lon"], joined_df["lat_centroids"], joined_df["lon_centroids"])
    joined_df["haversine"] = haversine_distance(joined_df["lat"], joined_df["lon"], joined_df["lat_centroids"], joined_df["lon_centroids"])
    print("Manhattan Distance")
    print(joined_df.groupby("zones")["manhattan"].sum())
    print("Haversine Distance")
    print(joined_df.groupby("zones")["haversine"].sum())
    return(joined_df["manhattan"].sum(), joined_df["haversine"].sum())

def tsp_zone_distance(transformed_df, main_df):
    zones = ["D1", "D2", "D3", "D4", "D5", "D6"]
    unpivoted_df = pd.melt(transformed_df, id_vars=['Id_Cliente'], var_name= "zones")
    unpivoted_df = unpivoted_df[unpivoted_df.value>0]
    joined_df = unpivoted_df.merge(main_df[["Id_Cliente", "lat", "lon"]], on="Id_Cliente", how = "left")
    total_distance = []
    for zone in zones:
        fitness_coords = mlrose.TravellingSales(coords = joined_df[joined_df["zones"]==zone][["lat", "lon"]].to_numpy())
        problem_fit = mlrose.TSPOpt(length = joined_df[joined_df["zones"]==zone].shape[0], fitness_fn = fitness_coords,
                            maximize=False)
        best_state, best_fitness = mlrose.genetic_alg(problem_fit, random_state = 12)
        total_distance.append(best_fitness)
    print(total_distance)
    return(sum(total_distance))