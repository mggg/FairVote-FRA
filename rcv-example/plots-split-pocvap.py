
import jsonlines
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
from pathlib import Path
import json
import math
import pandas as pd
import us
from evaltools.geometry import invert
import warnings
import sys

# Ward off warnings.
warnings.simplefilter("ignore", UserWarning)

# Set some pathing.
chains = Path("./output/chains/")

# Get the location and some info about the districts (and their magnitudes).
location = us.states.lookup(sys.argv[-1].title(), field="name")
with open("groupings.json") as r: grouping = json.load(r)[location.name.lower()]
optimal = grouping["optimal"]

# Set plot properties.
lw = 1.5
boxprops = {
    "patch_artist": True,
    "boxprops": dict(lw=lw, facecolor="None"),
    "showfliers": False,
    "medianprops": dict(color="k", lw=lw),
    "whiskerprops": dict(lw=lw),
    "capprops": dict(lw=lw),
    "zorder": 0,
    "whis": (1, 99)
}

# Read in the data for neutral and tilted ensembles.
with jsonlines.open(chains/f"{location.name.lower()}/neutral.jsonl") as r:
    neutral = [p for p in r]

with jsonlines.open(chains/f"{location.name.lower()}/tilted.jsonl") as r:
    tilted = [p for p in r]

# Get some information about the statewide CVAP shares.
nationwide = pd.read_csv("./data/demographics/nationwide.csv")
nationwide = nationwide.set_index("ABBR")
POCVAPSTATEWIDE = nationwide.loc[location.abbr, "POCVAP20%"]

# Now, get values for each of the magnitudes to create split plots. This enables
# us to re-use data.
boxes = {
    "neutral": {3: {}, 4: {}, 5: {}},
    "tilted": {3: {}, 4: {}, 5: {}}
}

# Set label and ticklabel font properties.
lfp = FontProperties(family="Playfair Display")
lfd = {"fontproperties": lfp}

tlfp = FontProperties(family="CMU Serif", size=8)
tlfd = {"fontproperties": tlfp}

# Create figures, left and right axes, and storage for handles.
fig, (left, right) = plt.subplots(1, 2, figsize=(7, 3))
handles = {3: set(), 4: set(), 5: set()}

# Iterate over the combinations of collections and names.
for collection, bias, axes in zip([neutral, tilted], ["neutral", "tilted"], [left, right]):
    for number, magnitude in zip(optimal, [3, 4, 5]):
        boxes[bias][magnitude] = { i: [] for i in range(number) }

    # For each of the plans in the collection, get the districts of each
    # magnitude; then, split the data into respective buckets according to
    # district magnitude.
    for plan in collection:
        # Get the number of members per district, then get their statistics.
        magnitudes = invert(plan["MAGNITUDE"])
        for magnitude, names in magnitudes.items():
            POCVAP = "POCVAP20" if plan.get("POCVAP20") else "POCVAP"
            VAP = "VAP20" if plan.get("POCVAP20") else "VAP"
            
            values = list(sorted(
                plan[POCVAP][district]/plan[VAP][district]
                for district in names
            ))

            # Add each of these to the boxes!
            for i, value in enumerate(values):
                boxes[bias][magnitude][i].append(value)

    # Create split plots; these are organized so that boxes are ordered
    # first by magnitude, *then* by percentage.
    seq = []
    for mag in [3, 4, 5]:
        for pct in boxes[bias][mag].values():
            if pct: seq.append(pct)

    # Change the boxplot witdth based on location.
    boxprops.update({
        "widths": 1/3 if location not in {us.states.MD, us.states.MA} else 1/4
    })

    axes.boxplot(seq, **boxprops)
    axes.set_ylim(bottom=0, top=math.ceil(max(max(b) for b in seq)/0.1)/10)
    
    # Add demarcation points for different district magnitudes.
    lineprops = { "ls": ":", "alpha": 1/2, "color": "k" }
    textprops = {"ha": "center", "va": "center", "zorder": 10, "fontproperties": lfp}

    # Do a bit of math to uniformly place text labels and titles.
    locs = axes.get_yticks()
    interval = locs[1]-locs[0]
    intervals = len(locs)-1
    basemultiplier = -3/5
    rescale = (intervals+1)/5
    scaledmultiplier = basemultiplier*rescale
    y = scaledmultiplier*interval if min(locs) == 0 else min(locs)-scaledmultiplier*interval

    # First, check if we have any 3-member districts.
    if optimal[0]:
        xlocs = axes.get_xticks()
        l, r = xlocs[0]-1/2, optimal[0] + 1/2
        mid = ((l+r)/2, y)
        axes.hlines(1/4, l, r, color="b", alpha=1/2)
        axes.axvline(r, **lineprops)
        axes.text(*mid, "3-member", **textprops)

        # Create a legend handle.
        handles[3].add(Line2D([0],[0], color="b", label="1 of 3 seats"))
    
    # Next, any four-member districts.
    if optimal[1]:
        l, r = optimal[0] + 1/2, optimal[0] + optimal[1] + 1/2
        mid = ((l+r)/2, y)
        axes.hlines(1/5, l, r, color="r", alpha=1/2)
        axes.axvline(r, **lineprops)
        axes.text(*mid, "4-member", **textprops)

        # Create a legend handle.
        handles[4].add(Line2D([0],[0], color="r", label="1 of 4 seats"))

    # Lastly, five-member districts.
    if optimal[2]:
        l, r = optimal[0] + optimal[1] + 1/2, sum(optimal) + 1/2
        mid = ((l+r)/2, y)
        axes.hlines(1/6, l, r, color="y", alpha=1/2)
        axes.text(*mid, "5-member", **textprops)

        # Create a legend handle.
        handles[5].add(Line2D([0],[0], color="y", label="1 of 5 seats"))

    # Add a proportionality line.
    axes.axhline(POCVAPSTATEWIDE, color="k", alpha=1/2)

    # Reset x- and y-labels.
    labels = list(pd.core.common.flatten(list(range(1, g+1)) for g in optimal))
    axes.set_xticklabels(labels, fontdict=tlfd)

    # Set x- and y-ticks.
    locs, labels = axes.get_yticks(), axes.get_yticklabels()
    axes.set_yticklabels([str(int(loc*100)) + r"%" for loc in locs], fontdict=tlfd)

    # Set titles.
    # axes.set_title(bias.capitalize(), y=-7/24)

# Create legend with custom handles for side-by-side figs.
handles = [handle.pop() for handle in handles.values() if handle]
handles.append(Line2D([0],[0], color="k", alpha=1/2, label="Proportionality"))


if location not in {us.states.MD, us.states.MA}:
    fig.legend(
        loc="upper center", prop=lfp, title="RCV Thresholds",
        borderaxespad=0, bbox_to_anchor=(0.5, 1.05), ncol=len(seq)+1, handles=handles,
        title_fontproperties=lfp
    )
else:
    fig.legend(
        loc="upper center", prop=lfp, title="RCV Thresholds",
        borderaxespad=0, bbox_to_anchor=(0.5, 1.05), ncol=len(seq)+1, handles=handles,
        title_fontproperties=lfp
    )

# To file.
write = Path(f"./output/figures/{location.name.lower()}/")
if not write.exists(): write.mkdir()
plt.savefig(write / f"POCVAP-split.png", dpi=600, bbox_inches="tight")
