
from gerrychain.graph import Graph
import pandas as pd
import geopandas as gpd
from pathlib import Path
import json
import us

# Set the graph root path.
graphroot = Path("./data/graphs/")

# Set the list of columns we want to keep.
keep = [
    "TOTPOP20", "OTHERPOP20", "BVAP20", "NWBHVAP20", "NWBHPOP20", "APAPOP20",
    "OTHERVAP20", "AMINVAP20", "WPOP20", "NHPIPOP20", "2MOREPOP20", "APAMIVAP20",
    "BPOP20", "ASIANPOP20", "NHPIVAP20", "ASIANVAP20", "VAP20", "APBPOP20",
    "APAMIPOP20", "APAVAP20", "2MOREVAP20", "APBVAP20", "WVAP20", "DOJBVAP20",
    "AMINPOP20", "HVAP20", "POCVAP20"
]

# Make a container for all the dataframes.
nationrecords = [] # is on your side

# For each state, get some data from the graph.
for state in us.states.STATES:
    # Read in the graph.
    graphpath = graphroot/f"{state.name.lower()}.json"
    with open(graphpath) as r: G = json.load(r)

    # Create a dataframe from the list of records.
    rawdf = pd.DataFrame.from_records(G["nodes"])
    df = rawdf[keep]
    
    # Sum over everything and create a record.
    nationrecords.append({
        "ABBR": state.abbr,
        "STATE": state.name.title(),
        **df.sum().to_dict()
    })

# Make the nationwide dataframe!
nationwide = pd.DataFrame.from_records(nationrecords)

# Create some new columns.
nationwide["POCVAP20%"] = nationwide["POCVAP20"]/nationwide["VAP20"]

apportionment = pd.read_csv("./data/demographics/apportionment.csv")
apportionment = dict(zip(apportionment["STATE"], apportionment["REPRESENTATIVES"]))
nationwide["POCVAP20SEATS"] = nationwide["POCVAP20%"]*nationwide["STATE"].map(apportionment)

nationwide.to_csv("./data/demographics/nationwide.csv", index=False)
