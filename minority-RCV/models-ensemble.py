
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import jsonlines
import numpy as np
from pathlib import Path
from collections import Counter
import sys


"""
Creates side-by-side typical STV outcomes based on candidate pool.
"""

# Specify the jurisdiction, from which we read both the simulation results.
JURIS = sys.argv[-2]
BIAS = sys.argv[-1]
RESULTS_PATH = Path(f"./data/fairvote/output/{JURIS}-{BIAS}.jsonl")
FIGURES_PATH = Path(f"./data/fairvote/figures/{JURIS}/")
if not FIGURES_PATH.exists(): FIGURES_PATH.mkdir()

# Read in the simulation results.
with jsonlines.open(RESULTS_PATH) as results: RESULTS = [result for result in results]

# For each of these results, we want to get the range of *all* the scenarios for
# each of the candidate pools and get the ranges. Create a bucket to store the
# data we need to make the plots.
POOLS = ["P2"]
BUCKET = { p: {} for p in POOLS}
SCENARIOS = ["A", "B", "C", "D"]
MODELS = ["plackett-luce", "bradley-terry", "crossover", "cambridge"]

# What data do we care about?
GROUP = "POCVAP"
WHOLE = "VAP"
SEATS = "SEATS"
GROUPSHARE = GROUP + "%"

for pool in BUCKET:
    for result in RESULTS:
        # Get the rank-order of the districts in the current result based on
        # proportion of seats won.
        order = []

        for identifier, district in result.items():
            # Create a list of proportions which we can concatenate later.
            district["TOTAL"] = np.array([
                list(district[model][pool].values()) for model in MODELS
            ]).flat.copy()

            # Get the proportion of seats won.
            k = district["facts"][SEATS]
            district["PROPORTIONS"] = district["TOTAL"] / k

            # Get proportion of total seats won.
            district["PROPORTIONTOTAL"] = district["TOTAL"].sum() / (k*len(SCENARIOS)*len(MODELS))

            # Append the district to the ordering.
            order.append(district)

        # Sort the districts by minimum seat percentage. Then iterate over this
        # order, adding the districts to our bucket as we go.
        order = list(sorted(order, key=lambda d: d["facts"][GROUPSHARE]))

        for i, district in enumerate(order):
            if BUCKET[pool].get(i, False): BUCKET[pool][i].append(district)
            else: BUCKET[pool][i] = [district]

# Use LaTeX. Create figures and axes.
plt.rcParams.update({
    "text.usetex": True,
    # "text.latex.preamble": r'\usepackage[cm]{sfmath}',
    "font.family": "serif",
	"font.serif": "New Century Schoolbook"
})

# Set figure sizes.
figuresizes = {
    "maryland": (3, 3),
    "florida": (5, 3),
    "massachusetts": (3, 3),
    "texas": (5, 3),
    "illinois": (5, 3)
}

if len(POOLS) > 1: figs, axs = plt.subplots(len(POOLS))
else:
    figs, axs = plt.subplots(figsize=figuresizes[JURIS])
    figs = [figs]
    axs = [axs]

# Now that we have our ordered results, we can make plots. We want to plot the
# range of STV wins with the group's share of the population, so we make plots.
for pool, ax in zip(BUCKET, axs):
    # Get the sequences of STV shares and group population shares, then interleave
    # them.
    STVs = [
        list(np.array([district["PROPORTIONS"] for district in ordering]).flat)
        for ordering in BUCKET[pool].values()
    ]

    # Get the average group proportion for the district type.
    PROPORTIONS = [
        list(np.array([district["facts"][GROUPSHARE] for district in ordering]))
        for ordering in BUCKET[pool].values()
    ]

    # Set titles.
    # ax.set_title(f"Candidate Pool {pool.split('P')[1]}")

    # Set line widths.
    baselinewidth = 1
    seatsharewidth = 8
    pocvapsharewidth = 6

    # For each of the STVs, make a nice-looking plot.
    for x, (STV, proportions) in enumerate(zip(STVs, PROPORTIONS)):
        ax.vlines(x, ymin=0, ymax=1, color="k", lw=baselinewidth)
        ax.vlines(
            x, ymin=min(STV), ymax=max(STV), color="#d12455", lw=seatsharewidth,
            capstyle="round"
        )

        # Plot line caps.
        ax.hlines(0, xmin=x-1/10, xmax=x+1/10, color="k", lw=baselinewidth)
        ax.hlines(1, xmin=x-1/10, xmax=x+1/10, color="k", lw=baselinewidth)

        # Plot the group share.
        if len(Counter(proportions)) > 1:
            ax.vlines(
                x, ymin=min(proportions), ymax=max(proportions), color="#f0c93e",
                lw=pocvapsharewidth, capstyle="round"
            )
        else:
            ax.vlines(
                x, ymin=min(proportions)-1/100, ymax=min(proportions)+1/100, color="#f0c93e",
                lw=pocvapsharewidth, capstyle="round"
            )

    # Turn the axes off.
    ax.axis("off")

    # Plot an "axis" at -1.
    ax.vlines(-1, ymin=0, ymax=1, color="k", capstyle="round")
    ax.hlines(1, xmin=-11/10, xmax=-9/10, color="k", capstyle="round")
    ax.hlines(0, xmin=-11/10, xmax=-9/10, color="k", capstyle="round")
    ax.text(-12/10, 1, r"$100\%$", ha="right", va="center", fontsize=10)
    ax.text(-12/10, 0, r"$0\%$", ha="right", va="center", fontsize=10)

    # Create annotation arrows.
    """
    annotationfontsize = 10
    ax.text(
        (len(STVs))-1, -3/20, fr"higher {GROUP} share $\rightarrow$",
        fontsize=annotationfontsize, ha="right"
    )
    ax.text(
        0, -3/20, fr"$\leftarrow$ lower {GROUP} share", fontsize=annotationfontsize,
        ha="left"
    )
    """

# Create a legend.
handles = [
    Line2D([0], [0], color="#d12455", lw=4, solid_capstyle="round"),
    Line2D([0], [0], color="#f0c93e", lw=4, solid_capstyle="round")
]

labels = [
    "Potential Seat Shares",
    f"{GROUP} Shares"
]

plt.legend(
    handles=handles, labels=labels, loc="upper center", fontsize=10,
    bbox_to_anchor=(0.5, 1.2), borderaxespad=0, ncol=2
)

# Adjust plot spacing and turn axes off.
if len(POOLS) > 1: plt.subplots_adjust(hspace=1/3)

# Set plot-wide title.
# plt.suptitle(f"STV Simulations in {JURIS.capitalize()} (Ensemble)", fontsize=16, y=1.05)
# plt.show()
plt.savefig(FIGURES_PATH/f"{BIAS}-ensemble.png", bbox_inches="tight", dpi=600)
