
import jsonlines
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.font_manager import FontProperties
from pathlib import Path
import us
import pandas as pd
import numpy as np
from collections import Counter

from ModelingResult import ModelingResult, aggregate

# Get the location for the chain and the bias.
focus = { us.states.FL, us.states.IL, us.states.MA, us.states.MD, us.states.TX }

# Get the configuration for the state.
poc = pd.read_csv("./data/demographics/pocrepresentation.csv")
statewide = pd.read_csv("./data/demographics/summary.csv")

summary = statewide.merge(poc, on="STATE")\
    .set_index("STATE")\
    .to_dict("index")

# Two groups of states.
stateorders = [
    list(us.states.STATES),
    list(
        sorted(us.states.STATES, key=lambda s: summary.get(s.name.title())["POCVAP20%"])
    )
]
stateorders = list(zip(["alphabetical", "POCVAP"], stateorders))

# Create a mapping for configurations.
concentrations = {
    "A": [0.5]*4,               # Voters more or less agree on candidate order for each group.
    "B": [2,0.5,0.5,0.5],       # POC voters vary on POC candidate rank-order, but other groups agree.
    "C": [2]*4,                 # Voters don't agree on anything.
    "D": [0.5,0.5,2,2],         # POC voters agree, and white voters don't.
    # "E": [1]*4                  # No agreement or disagreement --- it's pandemonium.
}

# Are we doing low-turnout?
reducedturnout = True
turnoutsuffix = "-lowturnout" if reducedturnout else ""

for ordering, states in stateorders:

    # Bucket for seat totals.
    totals = {}

    for state in states:
        # Load the plans for the ensemble.
        output = Path("./output/")
        planpath = Path(output/f"results/{state.name.lower()}/")

        # Buckets for everything.
        ensembletypes = ["-lowturnout"]
        tilted = False
        ensembles = []

        # Get the record for the individual state.
        srecord = summary[state.name.title()]
        if srecord["REPRESENTATIVES"] < 3: continue

        try:
            # Bucket for ensembles.
            for ensembletype in ensembletypes:
                # Bucket for plans.
                plans = []

                representatives = summary.get(state.name.title())["REPRESENTATIVES"]

                # Check to see whether we have the complete results.
                tpath = planpath/f"{ensembletype}-0{ensembletype}.jsonl"
                
                for plan in range(50 if representatives > 5 else 1):
                    try:
                        districts = []
                        with jsonlines.open(planpath/f"neutral-{plan}-lowturnout.jsonl") as r:
                            for district in r:
                                districts.append([ModelingResult(**c) for c in district])
                        plans.append(districts)
                    except: continue

                ensembles.append(plans)
        except Exception as e:
            print(e)
            continue

        models = ["plackett-luce", "bradley-terry", "crossover", "cambridge"]
        concentrations = ["A", "B", "C", "D"]
        pools = [1, 2, 3, 4]

        # Sample size.
        subsample = 3

        # Aggregate results.
        # aggregate(ensembles[0], models, concentrations, subsample=subsample)
        r = aggregate(ensembles[0], models, concentrations, subsample=subsample)["all"]

        totals[state] = {
            seats/srecord["REPRESENTATIVES"]: count/len(r)
            for seats, count in dict(Counter(r)).items()
        }

    # Create subplots.
    fig, (left, right) = plt.subplots(1, 2)

    # Plotting! Here, we want to make circles at the appropriate height by summing
    # over the seat totals from the *plans*. Set some defaults, like the max radius
    # of the circles.
    r = 1/2
    cdefs = dict(
        linewidth=1/4,
        edgecolor="grey",
        zorder=1
    )

    mdefs = dict(
        marker="x",
        color="gold",
        s=3
    )

    # Split the results in half.
    items = list(totals.items())
    first = list(reversed(items[:len(items)//2]))
    second = list(reversed(items[len(items)//2:]))

    # Set label and ticklabel font properties.
    lfp = FontProperties(family="Playfair Display")
    lfd = {"fontproperties": lfp}
    fp = FontProperties(family="CMU Serif")

    for ax, chunk in zip([left, right], [first, second]):
        for y, (state, results) in enumerate(chunk):
            # First, get the record.
            srecord = summary[state.name.title()]

            # Next, plot the circles!
            for x, share in results.items():
                # Get the appropriate radius relative to the max radius.
                sr = r*np.sqrt(share)

                # Plot a circle!
                C = Circle((x*10, y+1), sr, facecolor="steelblue")
                ax.add_patch(C)

                # Add a scatter point!
                ax.scatter(srecord["POCVAP20%"]*10, y+1, marker="s", s=14, color="k")
                ax.scatter(srecord["POCVAP20%"]*10, y+1, marker="s", s=12, color="gold")
                ax.scatter(
                    (srecord["POCREPRESENTATIVES"]/srecord["REPRESENTATIVES"])*10,
                    y+1, marker="o", s=14, color="k"
                )
                ax.scatter(
                    (srecord["POCREPRESENTATIVES"]/srecord["REPRESENTATIVES"])*10,
                    y+1, marker="o", s=12, color="red"
                )


        # Set plot limits.
        ax.set_ylim(0, y+2)
        ax.set_xlim(-1, 11)
        ax.set_aspect("equal")

        # Set ticks and labels?
        ax.set_yticks(range(1, y+2), labels=[s.abbr for s, _ in chunk])

        xlocs = list(ax.get_xticks())[1:-1]
        ax.set_xticks(xlocs, [str(int(x*10)) + "%" for x in xlocs])

        # Now go through and set font properties? this is bullshit
        for tickset in [ax.get_yticklabels()]:
            for tick in tickset:
                tick.set_fontproperties(lfp)

        for tickset in [ax.get_xticklabels()]:
            for tick in tickset:
                tick.set_fontproperties(fp)

        # Set titles.
        ax.set_xlabel("Seat Share", fontdict=lfd)

    fig.tight_layout()
    # plt.show()
    plt.savefig(f"./output/figures/nationwide-{ordering}-lowturnout.png", dpi=600, bbox_inches="tight")
    plt.clf()
