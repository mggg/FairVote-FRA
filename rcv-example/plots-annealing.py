
import jsonlines
import json
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from accept import logicycle
import sys
import us
from AnnealingConfiguration import AnnealingConfiguration
from pathlib import Path

# Use LaTeX.
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Playfair Display",
    "mathtext.fontset": "cm"
})

# Get location information.
location = us.states.lookup(sys.argv[-1].title(), field="name")
state = location.name.lower()

# Get the configuration.
with open("config.json") as r: configurations = json.load(r)
configuration = AnnealingConfiguration(**configurations[state])

# Are we testing?
TEST = False

if TEST:
    with open(f"./output/chains/{state}/neutral-seats.json") as r: nseats = json.load(r)
    with open(f"./output/chains/{state}/neutral-preference.json") as r: npreference = json.load(r)
    with open(f"./output/chains/{state}/tilted-seats.json") as r: seats = json.load(r)
    with open(f"./output/chains/{state}/tilted-anneal.json") as r: differential = json.load(r)
    with open(f"./output/chains/{state}/tilted-preference.json") as r: preference = json.load(r)

    # Pick the starting and stopping points.
    start = 0
    stop = len(seats)

else:
    # Load up the data.
    with jsonlines.open(f"./output/chains/{state}/tilted.jsonl") as r:
        plans = list(r)
        seats = [p["SEATS"] for p in plans]
        preference = [p["PREFERENCE"] for p in plans]
        differential = [p["ANNEAL"] for p in plans]

    with jsonlines.open(f"./output/chains/{state}/neutral.jsonl") as r:
        plans = list(r)
        nseats = [p["SEATS"] for p in plans]
        npreference = [p["PREFERENCE"] for p in plans]

    # Pick the starting and stopping points.
    start = len(seats)-(5*len(seats)//configuration.cycles)
    stop = len(seats)

# Create a three-column figure.
figure, (scores, energy, schedule) = plt.subplots(3, 1, figsize=(3,5))

# Plot the annealed scores over time.
scores.plot(range(len(seats)), seats, color="b", linewidth=1/4, label="seats (tilted)")
# scores.plot(range(len(nseats)), nseats, color="y", linewidth=1/4, label="seats (neutral)")
scores.plot(range(len(preference)), preference, color="g", linewidth=1/4, label="partial seats (tilted)")
# scores.plot(range(len(npreference)), npreference, color="m", linewidth=1/4, label="partial seats (neutral)")

# Set limits.
scores.set_xlim(start, stop)
# scores.legend(fontsize=8, loc="center right", ncol=1)

# Labels and title for annealed.
scores.set_xlabel(r"$t$", fontfamily="Playfair Display")
scores.set_ylabel("statewide thresholds")
scores.set_title("scores")

# Plot the energy of the plans seen.
medianenergy = np.median(differential)
energy.plot(range(len(differential)), differential, color="r", linewidth=1/4)
energy.set_xlim(start, stop)
energy.set_xlabel(r"$t$")
energy.set_ylabel(r"$e^{-\beta(t)\cdot\Delta J}$")
energy.set_title("energy differentials")

# Plot the temperature over time.
ITERATIONS = len(seats)
temp = logicycle(
    configuration.max,
    configuration.growth,
    ITERATIONS,
    configuration.midpoint,
    cold=configuration.cold,
    cycles=configuration.cycles
)

X = np.linspace(start=start, stop=stop, num=int((stop-start)*10))
Y = np.array([-temp(x) for x in X])

# Scatter based on values.
plt.scatter(X, Y, c=cm.RdBu(-Y/-Y.min()), edgecolor="none", s=1)
schedule.set_xlim(start, stop)
schedule.set_ylim(bottom=-(configuration.max+1), top=1/2)
schedule.set_xlabel(r"$t$")
schedule.set_ylabel(r"$-\beta(t)$")
schedule.set_title("annealing schedule")

figure.tight_layout()

write = Path(f"./output/figures/{state}")
if not write.exists(): write.mkdir()
plt.savefig(write/"annealing.png", dpi=600)
