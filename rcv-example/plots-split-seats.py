
import jsonlines
import matplotlib.pyplot as plt
from pathlib import Path
import json
import pandas as pd
import us
import sys
import seaborn as sns
from matplotlib.font_manager import FontProperties
from collections import Counter

# Set some pathing.
chains = Path("./output/chains/")

# Get the location and some info about the districts (and their magnitudes).
location = us.states.lookup(sys.argv[-1])
with open("groupings.json") as r: grouping = json.load(r)[location.name.lower()]
optimal = grouping["optimal"]

# Set plot properties.
lw = 1.5
boxprops = {
    "patch_artist": True,
    "boxprops": dict(lw=lw, facecolor="#0099cd"),
    "showfliers": False,
    "medianprops": dict(color="#fcd20e", lw=lw),
    "whiskerprops": dict(lw=lw),
    "whis": (1, 99),
    "capprops": dict(lw=lw),
    "zorder": 0,
    "widths": 1/3
}

# Are we testing?
TEST = False

# Get statewide data.
nationwide = pd.read_csv("./data/demographics/nationwide.csv")
nationwide = nationwide.set_index("STATE")

# Read in the data for neutral and tilted ensembles.
if TEST:
    with open(chains/f"{location.name.lower()}/neutral-seats.json") as r:
        neutral = json.load(r)
    
    with open(chains/f"{location.name.lower()}/tilted-seats.json") as r:
        tilted = json.load(r)

else:
    with jsonlines.open(chains/f"{location.name.lower()}/neutral.jsonl") as r:
        neutral = [p for p in r]

    with jsonlines.open(chains/f"{location.name.lower()}/tilted.jsonl") as r:
        tilted = [p for p in r]

    neutral = [plan["SEATS"] for plan in neutral]
    # tiltedseats = [sum(plan["SEATS"].values()) for plan in tilted]
    tilted = [plan["SEATS"] for plan in tilted]

# Get the statewide POCVAP share and seat share.
row = nationwide.loc[location.name.capitalize()]
POCVAPSEATSHARE = row["POCVAP20SEATS"]

# Create a boxplot.
fig, (lower, upper) = plt.subplots(2, 1, figsize=(1.5, 3))
upperhist = sns.histplot(neutral, discrete=True, ax=upper)
lowerhist = sns.histplot(tilted, discrete=True, ax=lower)

print(dict(Counter(neutral)))
print(dict(Counter(tilted)))

# Set some properties for the tick labels.
fp = FontProperties(family="CMU Serif", size=8)

# Set upper and lower ticks.
for ax in [upper, lower]:
    locs = ax.get_xticks()
    ax.set_xticks(locs, [str(int(l)) for l in locs], fontsize=8)

    locs = ax.get_yticks()
    ax.set_yticks(locs, [str(int(l)) for l in locs], fontsize=8)
    ax.set_ylabel(None)

    # Now go through and set font properties? this is bullshit
    for tickset in [ax.get_xticklabels(), ax.get_yticklabels()]:
        for tick in tickset:
            tick.set_fontproperties(fp)

# Set titles.
upper.set_title("Neutral", fontfamily="Playfair Display")
lower.set_title("Tilted", fontfamily="Playfair Display")

# To file.
write = Path(f"./output/figures/{location.name.lower()}/")
if not write.exists(): write.mkdir()

fig.tight_layout()
plt.savefig(write / f"POCVAP-seats.png", dpi=600,)
