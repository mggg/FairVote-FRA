
import us
import json

# Read in all the graphs and create a POCVAP column, subtracting WVAP from
# VAP.

for state in us.states.STATES:
    # Read the dual graph as json.
    with open(f"./data/graphs/{state.name.lower()}.json") as r: G = json.load(r)

    # Create a POCVAP column.
    for vertex in G["nodes"]:
        # Calculate the POCVAP and assert some stuff.
        pocvap = vertex["VAP20"] - vertex["WVAP20"]
        assert pocvap >= 0
        assert pocvap <= vertex["VAP20"]

        # Adjoin to the vertex!
        vertex["POCVAP20"] = pocvap
    
    # Write back to file.
    with open(f"./data/graphs/{state.name.lower()}.json", "w") as w: json.dump(G, w)
