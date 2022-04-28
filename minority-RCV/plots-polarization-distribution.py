
import json
import matplotlib.pyplot as plt
import us
import math
import numpy as np

from ModelingConfiguration import ModelingConfiguration


# Set focus states.
focus = { us.states.FL, us.states.IL, us.states.MA, us.states.MD, us.states.TX }

# Set ensemble types.
ensembletypes = ["neutral", "tilted"]

# Read in the data.
with open("configurations.json") as r: configurations = json.load(r)

# Go through and parse stuff out.
configurations = {
    state: configurations[state.name.lower()]
    for state in focus
}

# Get the things.
for state, configs in configurations.items():
    tilted, neutral = [], []

    for ensemble, ensembletype in zip([neutral, tilted], ensembletypes):
        for plan in configs[ensembletype]:
            thresholds = 0

            for district, C in enumerate(plan):
                first = ModelingConfiguration(**C.pop())
                thresholds += math.floor(first.pocshare/(1/(first.seats+1)))

            ensemble.append(thresholds)

    # Get counts.
    n = np.unique(neutral, return_counts=True)
    t = np.unique(tilted, return_counts=True)

    # Bars!
    n = [[l-1/8 for l in n[0]] ,n[1]]
    t = [[l+1/8 for l in t[0]], t[1]]

    fig, hist = plt.subplots()

    nc, tc = "mediumpurple", "mediumseagreen"
    histdefs = dict(edgecolor="k", align="center", linewidth=1, alpha=1/2, width=3/8)
    hist.bar(*n, color=nc, **histdefs)
    hist.bar(*t, color=tc, **histdefs)

    hist.set_title(state.name.capitalize())

    plt.savefig(
        f"./output/figures/polarization-distributions/{state.abbr}-green-and-green.png",
        dpi=600, bbox_inches="tight"
    )
