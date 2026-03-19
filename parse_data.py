import pandas as pd
import os 

adjacency_matrix = os.path.join(os.path.dirname(__file__),"data","adjacency_matrix_correct.parquet")
aero_anac = os.path.join(os.path.dirname(__file__), "data", "aero_anac_2017_2023.parquet")
fluvi_road = os.path.join(os.path.dirname(__file__),"data","fluvi_road_ibge.parquet")

df = pd.read_parquet(adjacency_matrix)
print(df.head())

#incorporate a bayesian hierarchical model, with inputs of
#our datatypes, as well as cases from our last project