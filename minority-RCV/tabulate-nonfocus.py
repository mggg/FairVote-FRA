
import pandas as pd
import us
import jsonlines
import json
import numpy as np
from collections import Counter
from pathlib import Path

from ModelingResult import ModelingResult

# Read in statewide summaries and merge.
statewide = pd.read_csv("./data/demographics/summary.csv")
polarization = pd.read_csv("./data/demographics/polarization.csv")
summary = statewide.merge(polarization, on="STATE")

# Now, for each of the non-focus states, we want to discuss the breakdown of the
# seats, how many seats we'd expect PoC to win under a proportional system, and
# relate the results (in a 4x4 subtable maybe?)
subtable = summary[
    [
        "STATE", "TOTPOP20", "POCVAP20%", "REPRESENTATIVES", "POCVAP20SEATS",
        "3", "4", "5", "pp", "pw", "wp", "ww"
    ]
]

# Correct some typing.
subtable["TOTPOP20"] = subtable["TOTPOP20"].astype(int).apply(lambda s: f"{s:,}")
subtable["POCVAP20%"] = (subtable["POCVAP20%"]*100).round(1).astype(str) + "%"
subtable["POCVAP20SEATS"] = subtable["POCVAP20SEATS"].round(1)
subtable["pp"] = subtable["pp"].round(2)
subtable["pw"] = subtable["pw"].round(2)
subtable["wp"] = subtable["wp"].round(2)
subtable["ww"] = subtable["ww"].round(2)

# Change some names!
subtable = subtable.rename({
    "STATE": "State",
    "TOTPOP20": "Pop.",
    "POCVAP20%": r"POCVAP\%",
    "REPRESENTATIVES": "Seats",
    "POCVAP20SEATS": "POC Prop.",
    "3": "3-mem",
    "4": "4-mem",
    "5": "5-mem",
    "pp": r"$\pi_{cc}$",
    "pw": r"$\pi_{cw}$",
    "wp": r"$\pi_{wc}$",
    "ww": r"$\pi_{ww}$"
}, axis=1)

# List out the focus states.
focus = { us.states.IL, us.states.FL, us.states.MA, us.states.MD, us.states.TX }

# Index things by state name.
states = statewide.set_index("STATE").to_dict(orient="index")

# Create stuff for predictions.
predictions = {}

# For each of the states, get the results.
for s in us.states.STATES:
    if s in focus:
        predictions[s.name.title()] = ["-", "-"]
        continue

    # Get the path to the results. If it doesn't exist, continue; else, grab them.
    results = Path(f"./output/results/{s.name.lower()}/neutral.jsonl")

    if not results.exists():
        predictions[s.name.title()] = ["-", "-"]
        continue

    with jsonlines.open(results) as r: ensemble = [p for p in r]

    # Each plan is a dictionary which maps a district's index to the list of configurations
    # corresponding to that district. From there, we just need to order the districts
    # in the plan by POCVAP share -- since the POCVAP share of each district in the plan
    # is constant over all configurations, we can just sample the first one and then
    # order them.
    for plan in ensemble:
        for index, district in plan.items():
            plan[index] = [ModelingResult(**r) for r in district]


    # Now, for each of the plans, sort the results into different buckets based on
    # model type.
    results = []

    modelnames = ["plackett-luce", "bradley-terry", "crossover", "cambridge"]

    for plan in ensemble:

        for name in modelnames:
            # For each district, get the configurations corresponding to the models,
            # for which there should be four: one for each candidate pool. Then,
            # we get the statewide seat totals for each configuration and each
            # simulation.
            statewide = {
                id: [
                    c for c in configurations if c.model == name
                ]
                for id, configurations in plan.items()
            }

            # Create a numpy array whose columns are simulations and rows are
            # results from different districts.
            for pool in range(4):
                subtotals = np.array([
                    district[pool].pocwins for district in statewide.values()
                ])
                results.extend(list(np.sum(subtotals, axis=0)))

    # Now, for each of the models, get the counts of predictions.
    popular = list(dict(Counter(results).most_common()[:2]).keys())
    predictions[s.name.title()] = [
        "-" if i+1 > len(popular) else popular[i] for i in range(2)
    ]

subtable["First"] = subtable["State"].apply(lambda s: predictions[s][0])
subtable["Second"] = subtable["State"].apply(lambda s: predictions[s][1])

# Write the table to file.
subtable.to_csv("./output/nonfocus.csv", index=False)
