import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pulp
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from k_means_constrained import KMeansConstrained
#Add your cplex algo, do not try to change it
path_to_cplex = "/opt/ibm/ILOG/CPLEX_Studio1210/cplex/bin/x86-64_linux/cplex"

def stops_gap(per):
    upper = int(663 + (663 * per))
    lower = int(662 - (662 * per))
    return(upper, lower)

def items_gap(per):
    upper = int(9067 + (9067 * per))
    lower = int(9066 - (9066 * per))
    return(upper, lower)

def main(input_path = "ubicaciones.csv", balance_deviations = [0.1, 0.15, 0.2, 0.3]):
    """
    Use different balance deviations to test complex lp function to minimize.
    Args:
        input_path : csv path with information regarding agencies, frequency, volume and coordinates
        balance_deviations = list with percentage of deviations, for calaculation refer to stops_gap and items_gap
    Returns:
        None
    """
    df = pd.read_csv(input_path)
    df.loc[df[df["Vol_Entrega"] == 0].index, "Vol_Entrega"] = 1

    zones = ["D1", "D2", "D3", "D4", "D5", "D6"]
    agencies = list("A" + df["Id_Cliente"].astype(str))
    vol_delivery = list(df["Vol_Entrega"])
    vol_stores = list(df["Vol_Entrega"]*df["Frecuencia"])
    frequency = list(df["Frecuencia"])
    stores_volume = dict(zip(agencies, vol_stores))
    stores_frequency = dict(zip(agencies, frequency))
    vol_delivery = dict(zip(agencies, vol_delivery))

    scaler = MinMaxScaler()
    fitted_scaler = scaler.fit(df[["lat", "lon"]])
    scaled_coordinates = fitted_scaler.transform(df[["lat", "lon"]])

    kmeans = KMeansConstrained(
            n_clusters=6,
            size_min=604,
            size_max=605,
            random_state=12,
            n_init=100,
            max_iter=200,
            n_jobs = -1)
    kmeans_values = kmeans.fit(scaled_coordinates)
    df["kmeans"] = list(kmeans.predict(scaled_coordinates))

    vectorized_lat_lon = df[["lat", "lon"]].to_numpy()
    cluster_centers = fitted_scaler.inverse_transform(kmeans.cluster_centers_)
    distance_matrix = cdist(cluster_centers, vectorized_lat_lon, metric= "cityblock")

    routes = [(z, a) for z in zones for a in agencies]
    distances = pulp.makeDict([zones, agencies], distance_matrix, 0)
    flow = pulp.LpVariable.dicts("Distribution", (zones, agencies), 0, None)
    using = pulp.LpVariable.dicts("BelongstoZone", (zones, agencies), 0, 1, pulp.LpInteger)

    for percentage in balance_deviations:
        prob = pulp.LpProblem("BrewingDataCup2020_" + str(percentage) , pulp.LpMinimize)
        prob += pulp.lpSum([distances[z][a] * flow[z][a]  for (z, a) in routes]) + pulp.lpSum([distances[z][a] * using[z][a]  for (z, a) in routes]), "totalCosts"
        stops_upper, stops_lower = stops_gap(percentage)
        distr_upper, distr_lower = items_gap(percentage)
        for z in zones:
            prob += pulp.lpSum([using[z][a] for a in agencies]) <= stops_upper, "SumStopsInZoneUpper %s"%z
            prob += pulp.lpSum([using[z][a] for a in agencies]) >= stops_lower, "SumStopsInZoneLower %s"%z
            prob += pulp.lpSum([flow[z][a] for a in agencies]) <= distr_upper, "SumDistrInZoneUpper %s"%z
            prob += pulp.lpSum([flow[z][a] for a in agencies]) >= distr_lower, "SumDistrInZoneLower %s"%z
        for z in zones:
            for a in agencies:
                prob += flow[z][a]-(100000*using[z][a]) <= 0
                prob += flow[z][a] <= vol_delivery[a]
        for a in agencies:
            prob += pulp.lpSum([flow[z][a] for z in zones]) >= stores_volume[a], "Distribution %s"%a
            prob += pulp.lpSum([using[z][a] for z in zones]) == stores_frequency[a], "FrequencyDistribution %s"%a
            
        prob.writeLP("lp_files/milp_brewing_" + str(percentage) + ".lp")
        solver = pulp.CPLEX_CMD(path=path_to_cplex)
        prob.solve(solver)
        print("Estado: ", pulp.LpStatus[prob.status])
        print("Total Cost: ", pulp.value(prob.objective))

        final_df = pd.DataFrame(columns = ["D1", "D2", "D3", "D4", "D5", "D6"], index=(range(1, 3626)))
        final_distr = dict()
        for v in prob.variables():
            if (v.name).find("BelongstoZone_")==0:
                if v.varValue > 0:
                    dist = v.name[14:]
                    zone = dist[:2]
                    id_cliente = int(dist[4:])
                    final_df.loc[id_cliente, zone] = 1
            
        final_df.fillna(0, inplace = True)
        final_df = final_df.astype(int).reset_index().rename(columns = {"index":"Id_Cliente"})
        final_df.to_csv("lp_solutions/cplex_opt_" + str(percentage) + "_" + str(pulp.value(prob.objective)) + ".csv", header = True, index = False)

if __name__== "__main__":
    main()