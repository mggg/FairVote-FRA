
import jsonlines
import pandas as pd
import us

from accept import districts as n_districts

# Sample these locations.
states = {
    us.states.MD: 8,
    us.states.MA: 9,
    us.states.TX: 38,
    us.states.FL: 28,
    us.states.IL: 17
}

for bias in ["neutral", "tilted"]:
    # For each of the locations, create JSON objects for each of the plans and their
    # statistics.
    for state, k in states.items():
        # Iterate over each of the plans, recording data as we go.
        records = []
        with jsonlines.open(f"./output/chains/{state.name.lower()}/{bias}.jsonl") as data:
            # Calculate the number of members per district.
            for record in data:
                record["POPULATION"] = record["population"]
                del record["population"]

                # Get the total population.
                N = sum(record["POPULATION"].values())

                record["DISTRICTS"] = {
                    d: n_districts(N, k)(record["POPULATION"][d])
                    for d in record["POPULATION"].keys()
                }

                # Now, we invert the dictionary.
                inverted = {}
                for district in record["POPULATION"].keys():
                    inverted[district] = {}

                    for updater in ["VAP", "POCVAP", "POPULATION", "DISTRICTS"]:
                        inverted[district][updater] = record[updater][district]
                        inverted[district]["PREFERENCE"] = record["PREFERENCE"]
                        
                records.append(inverted)
                    
        
        # Write all the records to file.
        with jsonlines.open(f"./output/records/{state.name.lower()}-{bias}.jsonl", mode="w") as w:
            w.write_all(records)