
from gerrychain.graph import Graph
import geopandas as gpd
import pandas as pd
import us

states = [
    us.states.MD,
    us.states.MA,
    us.states.TX,
    us.states.FL,
    us.states.IL
]

# For each of the locations, generate a dual graph.
for state in states:
    gdf = gpd.read_file(f"./data/geometries/{state.abbr.lower()}_vtd/")
    gdf["GEOID20"] = gdf["GEOID20"].astype(str)
    gdf = gdf.set_index("GEOID20")

    # Create the dual graph.
    g = Graph.from_geodataframe(gdf)

    # Write to file.
    g.to_json(f"./data/graphs/{state.name.lower()}.json")
