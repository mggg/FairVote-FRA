
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# Read in polarization and summary data.
polarization = pd.read_csv("./data/demographics/polarization.csv")
summary = pd.read_csv("./data/demographics/summary.csv")
demographics = summary.merge(polarization, on="STATE")

# How should we go about doing this?
demographics["WVAP20%"] = 1-demographics["POCVAP20%"]

# Create subplots, and make sure we're using LaTeX!
plt.rcParams.update({"font.family": "Playfair Display"})
fig, ax = plt.subplots()

# Set label and ticklabel font properties.
lfp = FontProperties(family="Playfair Display", size=14)
lfd = {"fontproperties": lfp}

tlfp = FontProperties(family="CMU Serif", size=12)
tlfd = {"fontproperties": tlfp}

# Set limits?
plt.ylim(0, 1)
plt.xlim(0, 1)

# Set ticks and tick labels!
locs, labels = plt.xticks()
plt.xticks(locs, [str(int(round(100*p))) + "%" for p in locs])

locs, labels = plt.yticks()
plt.yticks(locs, [str(int(round(100*p))) + "%" for p in locs])


# Set tick labels!
for labelset in [ax.get_xticklabels(), ax.get_yticklabels()]:
    for label in labelset: label.set_fontproperties(tlfp)

# Set label defaults.
labels = dict(
    fontsize=5, ha="center", va="center", color="white", weight="heavy"
)

# Plot labels!
for name, x, y in zip(demographics["ABBR"], demographics["POCVAP20%"], demographics["pp"]):
    ax.scatter(x, y, c="mediumpurple", s=100, zorder=10)
    ax.annotate(name, xy=(x,y), **labels, zorder=10)

# Plot labels!
for name, x, y in zip(demographics["ABBR"], demographics["WVAP20%"], demographics["ww"]):
    ax.scatter(x, y, c="mediumseagreen", s=100, zorder=11)
    ax.annotate(name, xy=(x,y), **labels, zorder=11)

# Label axes!
ax.set_xlabel("VAP Share", font=lfp, labelpad=10)
ax.set_ylabel("Support for CoC", font=lfp, labelpad=10)

fig.tight_layout()
plt.savefig("./output/figures/polarization.png", dpi=600, bbox_inches="tight")
