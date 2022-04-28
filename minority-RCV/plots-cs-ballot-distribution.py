
import pickle as pkl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

with open("./data/databases/Cambridge_09to17_ballot_types.p", "rb") as r:
    ballots = pkl.load(r)

# Get all the possible ballot types.
rawtypes = {
    "".join(k): v
    for k, v in ballots.items()
}

orderedtypes = {
    k: v for k, v in sorted(rawtypes.items(), key=lambda t: t[1], reverse=True)
}

total = sum(orderedtypes.values())
labels = list(orderedtypes.keys())[:30]
locs = list(range(len(labels)))[:30]
heights = list([v/total for v in orderedtypes.values()])[:30]

# Set label and ticklabel font properties.
lfp = FontProperties(family="Playfair Display")
lfd = {"fontproperties": lfp}
fp = FontProperties(family="CMU Serif")

# Create the bar plot!
_, ax = plt.subplots(figsize=(7,2))
plt.bar(locs, heights)

# Set x-axis tick labels.
plt.xticks(ticks=locs, labels=labels, rotation=90)

# Now go through and set font properties? this is bullshit
for tickset in [ax.get_yticklabels()]:
    for tick in tickset:
        tick.set_fontproperties(fp)

for tickset in [ax.get_xticklabels()]:
    for tick in tickset:
        tick.set_fontproperties(lfp)

# Set axis labels.
plt.ylabel("fraction of total ballots", fontdict=lfd)
plt.xlabel("ballot type", fontdict=lfd)

plt.savefig("./output/figures/ballot-types.png", dpi=600, bbox_inches="tight")
